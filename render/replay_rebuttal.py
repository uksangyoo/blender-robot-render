"""Roll out a VibeAct rebuttal-task policy and dump scene data for Blender.

Covers the three CoRL-rebuttal tasks in ``POLICY/rebuttal_tasks_envs.py``
(phone_flip, screwdriver_turn, card_pickup). Same output triple as
``replay_peg_climb.py`` (``scene.json`` / ``cameras.json`` / ``traj.npz``), with
two differences:

* The run's own ``method.env`` is replayed before the env is built
  (``eval_rebuttal_tasks.load_method``), so the rollout uses the tactile
  encoding, reward version and task options the policy was trained with.
* The task objects (phone slab, screwdriver, card) are scaled visual meshes the
  env generates at build time, so there is no source file to point Blender at.
  They are written out as OBJ straight from the compiled model (vertices already
  scaled and aligned), and their geoms keep the compiled pose.

``traj.npz`` also carries a per-step task-progress trace and the first success
step, which ``render_rebuttal_strip.py`` uses to pick frames along the task.

Run under the **lerobot** env from the vibereact repo root::

    PYTHONPATH=$PWD MUJOCO_GL=egl conda run -n lerobot --no-capture-output python \\
        ~/Projects/blender-robot-render/render/replay_rebuttal.py \\
        --task card_pickup --run-dir output/policies/norm_v5/card_pickup_vibeact_seed1 \\
        --out ~/Projects/blender-robot-render/outputs/rebuttal/card_pickup
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_peg_climb import _quat_conj, _quat_mul, _quat_rotate  # noqa: E402

logger = logging.getLogger("replay_rebuttal")

def _object_motion(env, info):
    """Generic progress for the paper tasks: how far the held object has moved
    (mm) plus rotated (deg) from its pose at the first step."""
    import mujoco
    u = env
    bid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    pos, quat = u.data.xpos[bid].copy(), u.data.xquat[bid].copy()
    if not hasattr(u, "_render_ref"):
        u._render_ref = (pos, quat)
    p0, q0 = u._render_ref
    dot = min(1.0, abs(float(np.dot(q0, quat))))
    return 1000.0 * float(np.linalg.norm(pos - p0)) + float(np.degrees(2.0 * np.arccos(dot)))


# Per-task progress trace: what "further along the task" means, read from info.
PROGRESS = {
    "phone_flip": lambda env, info: abs(float(info.get("flip_dev_deg", 0.0))) + 180.0 * info.get("flips", 0),
    "screwdriver_turn": lambda env, info: float(info.get("best_revs_cw", 0.0)),
    "card_pickup": lambda env, info: float(info.get("card_h", 0.0)),
    # Paper tasks: each env's own measure where it reports one.
    "cracker_climb": lambda env, info: float(info.get("climb_best", 0.0)),
    "peg_climb": lambda env, info: float(info.get("climb_best", 0.0)),
    "peg_in_hole": _object_motion,
    "cube_rotation": lambda env, info: float(info.get("cw_revs_total", 0.0)),
    "hex_nut_fingers": lambda env, info: float(info.get("nut_revs_cw", 0.0)),
}


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def build_scene_graph(model, out_dir: Path) -> dict:
    import mujoco
    from POLICY._env_utils import _LEAP_ASSETS, _XARM_ASSETS

    type_to_str = {getattr(mujoco.mjtGeom, n): n.replace("mjGEOM_", "").lower()
                   for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")}
    search_dirs = [Path(_LEAP_ASSETS), Path(_XARM_ASSETS)]
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    def resolve(mid: int):
        """(file, from_compiled). Robot meshes -> source file; others -> OBJ dump."""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh_{mid}"
        stem = name.split("/")[-1]
        for d in search_dirs:
            for ext in (".obj", ".stl", ".ply"):
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    return str(cand.resolve()), False
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        p = mesh_dir / f"{stem}.obj"
        _write_obj(p, model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn])
        return str(p.resolve()), True

    bodies = []
    for bi in range(model.nbody):
        bodies.append({"index": bi,
                       "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or f"body_{bi}",
                       "geoms": []})
    n_emit = 0
    for gi in range(model.ngeom):
        group = int(model.geom_group[gi])
        if group in (3, 4, 5):
            continue
        matid = int(model.geom_matid[gi])
        rgba = [float(x) for x in (model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[gi])]
        if rgba[3] <= 0.0:
            continue
        gtype = type_to_str.get(int(model.geom_type[gi]), "unknown")
        geom_pos = tuple(float(x) for x in model.geom_pos[gi])
        geom_quat = tuple(float(x) for x in model.geom_quat[gi])
        mesh_file = mesh_name = None
        user_pos, user_quat = geom_pos, geom_quat
        mesh_id = int(model.geom_dataid[gi])
        if gtype == "mesh" and mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            mesh_file, compiled = resolve(mesh_id)
            if not compiled:
                # Source files are unaligned: strip MuJoCo's internal alignment
                # (see replay.py:build_scene_graph).
                mq = tuple(float(x) for x in model.mesh_quat[mesh_id])
                mp = tuple(float(x) for x in model.mesh_pos[mesh_id])
                user_quat = _quat_mul(geom_quat, _quat_conj(mq))
                off = _quat_rotate(user_quat, mp)
                user_pos = tuple(geom_pos[i] - float(off[i]) for i in range(3))
        bi = int(model.geom_bodyid[gi])
        bodies[bi]["geoms"].append({
            "index": gi,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom_{gi}",
            "type": gtype,
            "size": [float(s) for s in model.geom_size[gi]],
            "local_pos": list(user_pos),
            "local_quat": list(user_quat),
            "rgba_geom": rgba,
            "material": (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
                         if matid >= 0 else None),
            "mesh_name": mesh_name,
            "mesh_file": mesh_file,
            "mesh_scale": [1.0, 1.0, 1.0],
            "group": group,
        })
        n_emit += 1
    # The object's collision primitives (hidden in the render) give its clean
    # local frame, which the figure uses to place markers that make motion
    # legible: a dark screen face on the phone, a stripe on the screwdriver.
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    obj_collision = []
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) == obj_body and int(model.geom_group[gi]) == 3:
            obj_collision.append({
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi),
                "type": type_to_str.get(int(model.geom_type[gi]), "unknown"),
                "size": [float(x) for x in model.geom_size[gi]],
                "local_pos": [float(x) for x in model.geom_pos[gi]],
                "local_quat": [float(x) for x in model.geom_quat[gi]],
            })
    logger.info("scene graph: %d bodies, %d visual geoms", model.nbody, n_emit)
    return {"bodies": bodies, "cameras": [], "object_collision": obj_collision}


def _make_env(task: str, run_dir: Path, seed: int):
    # A fresh env per rollout: domain randomization draws from a generator
    # seeded at construction, so reset(seed=...) alone does not reproduce an
    # episode.
    from POLICY.eval_rebuttal_tasks import COND_SLIP_CHANNELS, load_method
    from POLICY.train_peg import make_env
    meth = load_method(run_dir)
    recipe = (run_dir / "recipe.txt")
    if recipe.exists():
        # Paper-task runs (train_paper_render_policies.sh): encoding flags come
        # from the recipe the run was trained with.
        toks = recipe.read_text().split()
        get = lambda k, d: toks[toks.index(k) + 1] if k in toks else d
        feats = get("--slip-features", None)
        kw = dict(slip_encoding=get("--slip-encoding", "legacy"),
                  slip_channels=get("--slip-channels", "onset_binary_mag"),
                  slip_features=set(feats.split(",")) if feats else None,
                  expose_slip="--mask-contact" not in toks)
    else:
        cond = next(c for c in COND_SLIP_CHANNELS if run_dir.name.startswith(f"{task}_{c}_"))
        kw = dict(slip_encoding="legacy", slip_channels=COND_SLIP_CHANNELS[cond],
                  slip_features=None, expose_slip=True)
    fn = make_env(seed=seed, randomize=True, expose_slip=kw["expose_slip"], env_kind=task,
                  slip_history_len=meth["slip_history_len"], slip_reward_coef=0.0,
                  slip_encoding=kw["slip_encoding"], slip_recovery_coef=0.0, slip_smooth=False,
                  cracker_hard_dr=False, fixed_episode_length=True,
                  slip_channels=kw["slip_channels"], slip_features=kw["slip_features"])
    return fn()


def _rollout(env, policy, task: str, seed: int, capture: bool, deterministic: bool = True):
    base = env.unwrapped
    obs, _ = env.reset(seed=seed)
    frames, progress = [], []
    success_step = -1
    info = {}
    done = False
    t = 0
    import mujoco
    obj_b = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    palm_b = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, "leap_right/palm_lower")
    sep, upright = [], []
    while not done:
        if capture:
            frames.append((base.data.xpos.copy(), base.data.xquat.copy(), float(base.data.time)))
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, info = env.step(action)
        progress.append(PROGRESS[task](base, info))
        sep.append(float(np.linalg.norm(base.data.xpos[obj_b] - base.data.xpos[palm_b])))
        upright.append(float(base.data.xmat[obj_b].reshape(3, 3)[2, 2]))
        if success_step < 0 and info.get("success", False):
            success_step = t
        done = term or trunc
        t += 1
    if capture:
        frames.append((base.data.xpos.copy(), base.data.xquat.copy(), float(base.data.time)))
    upto = success_step + 1 if success_step >= 0 else len(sep)
    return {"success": bool(info.get("success", False)), "success_step": success_step,
            "max_sep": max(sep[:upto]) - sep[0] if sep else 0.0,
            "min_upright": min(upright[:upto]) if upright else 1.0,
            "dropped": bool(info.get("dropped", False)), "final_progress": progress[-1],
            "max_progress": max(progress), "progress": progress, "frames": frames}


def main() -> None:
    from stable_baselines3 import PPO

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(PROGRESS))
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--ckpt", default="final_model.zip")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 112)))
    p.add_argument("--prefer-upright", action="store_true",
                   help="among successes, pick the one whose object stays most upright")
    p.add_argument("--prefer-held", action="store_true",
                   help="among successes, pick the one whose object stays nearest the palm")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy instead of using its mean")
    p.add_argument("--min-success-step", type=int, default=30)
    p.add_argument("--target-step", type=int, default=150,
                   help="prefer the genuine success whose first success step is closest to this")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir.resolve()

    policy = PPO.load(str(run_dir / args.ckpt), device="cpu")
    logger.info("loaded %s", run_dir / args.ckpt)

    # Score seeds: a genuine, still-held success first; among those the one
    # whose success lands nearest --target-step, so the strip has a similar
    # pace across tasks.
    scores = []
    kept = {}
    for s in args.seeds:
        env = _make_env(args.task, run_dir, s)
        if args.stochastic:
            import torch
            torch.manual_seed(s)
        r = _rollout(env, policy, args.task, s, capture=True, deterministic=not args.stochastic)
        model = env.unwrapped.model
        env.close()
        # Rollouts do not replay bit-for-bit from a seed (render noise), so the
        # scored episode itself is the one exported.
        kept[s] = (r, model)
        logger.info("  seed %d: success=%s step=%d dropped=%s max_progress=%.3f max_sep=%.3f min_upright=%.3f",
                    s, r["success"], r["success_step"], r["dropped"], r["max_progress"], r["max_sep"], r["min_upright"])
        # Success inside the first MIN_SUCCESS_STEP steps is the object settling
        # at reset, not the policy doing the task: not a figure episode.
        genuine = r["success"] and not r["dropped"] and r["success_step"] >= args.min_success_step
        # With no genuine success anywhere, fall back to the episode that moves
        # the task furthest (the tie-break below is ignored for successes).
        if genuine and args.prefer_upright:
            # Among successes, the one whose object axis stays nearest vertical.
            tie = r["min_upright"]
        elif genuine and args.prefer_held:
            # Among successes, the one whose object stays closest to the palm
            # up to success: a climb, not the object pushed off the hand.
            tie = -r["max_sep"]
        elif genuine:
            tie = -abs(r["success_step"] - args.target_step)
        else:
            tie = r["max_progress"]
        scores.append((genuine, tie, s))
    scores.sort(reverse=True)
    best = scores[0][2]
    logger.info("best seed %d", best)

    r, model = kept[best]
    scene = build_scene_graph(model, args.out)
    np.savez_compressed(
        args.out / "traj.npz",
        timestamps=np.array([f[2] for f in r["frames"]], dtype=np.float64),
        body_pos=np.stack([f[0] for f in r["frames"]]).astype(np.float32),
        body_quat=np.stack([f[1] for f in r["frames"]]).astype(np.float32),
        body_names=np.array([b["name"] for b in scene["bodies"]]),
        progress=np.array([0.0] + r["progress"], dtype=np.float32),
        success_step=np.int64(r["success_step"] + 1 if r["success_step"] >= 0 else -1),
        success=np.bool_(r["success"]),
        seed=np.int64(best),
    )
    (args.out / "scene.json").write_text(json.dumps(scene, indent=1))
    (args.out / "cameras.json").write_text("{}")
    meta = {"task": args.task, "run_dir": str(run_dir), "ckpt": args.ckpt, "seed": best,
            "stochastic": bool(args.stochastic),
            "success": r["success"], "success_step": r["success_step"],
            "frames": len(r["frames"]), "max_progress": r["max_progress"]}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    logger.info("wrote %s: %s", args.out, meta)


if __name__ == "__main__":
    main()
