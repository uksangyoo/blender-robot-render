"""Roll out a trained PegClimb policy and dump portable scene data for Blender.

This is the PegClimb (xArm7 + LEAP hand + chips-can "peg") analogue of
``replay.py`` (which targets graph-as-policy LIBERO trials). Instead of
replaying a logged trial, it loads a trained PPO policy, rolls out a few
deterministic episodes, picks the best one (success, then max climb), and
writes the same ``scene.json`` / ``cameras.json`` / ``traj.npz`` triple that
``build_scene_pegclimb.py`` consumes.

Run under the **lerobot** conda env from the repo root::

    PYTHONPATH=/home/uyoo/vibereact MUJOCO_GL=egl \\
    conda run -n lerobot --no-capture-output python \\
        thirdparty/blender-robot-render/render/replay_peg_climb.py \\
        --policy output/policies/vibeact_paper_peg_climb_dense_seed3/best_success.zip \\
        --out thirdparty/blender-robot-render/outputs/peg_climb_seed3 \\
        --seeds 1 0 4 3 --randomize

Outputs into ``<out>/``:

* ``traj.npz``    : body_pos (T,NB,3), body_quat (T,NB,4 wxyz), body_names,
                    timestamps (T,), plus per-step peg climb for reference.
* ``scene.json``  : per-body / per-geom asset graph (mesh paths, local pose,
                    effective rgba, primitive sizes) for the *visual* geoms.
* ``cameras.json``: the scene_cam static pose + fovy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

logger = logging.getLogger("replay_peg_climb")


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz) — mirror render/replay.py so the mesh-internal
# alignment stripping is identical.
# ---------------------------------------------------------------------------

def _quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(q, v):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R @ np.asarray(v, dtype=np.float64)


# ---------------------------------------------------------------------------
# Mesh-file resolution. MuJoCo's compiled model keeps mesh *names* but not
# the source file paths (and mj_saveLastXML emits bare filenames relative to
# a meshdir that Blender can't see). We resolve names back to absolute paths
# by searching the known asset directories.
# ---------------------------------------------------------------------------

def _build_mesh_file_map(model):
    import mujoco
    from POLICY._env_utils import _XARM_ASSETS, _LEAP_ASSETS
    from POLICY.peg_climb_env import CHIPS_CAN_MESH_DIR, CHIPS_CAN_VISUAL_OBJ

    search_dirs = [Path(_LEAP_ASSETS), Path(_XARM_ASSETS)]
    exts = (".obj", ".stl", ".ply")

    def resolve(mesh_name: str):
        if mesh_name == "chips_can_visual_mesh":
            return (CHIPS_CAN_MESH_DIR / CHIPS_CAN_VISUAL_OBJ).resolve()
        stem = mesh_name.split("/")[-1]  # strip 'leap_right/' style prefixes
        for d in search_dirs:
            for ext in exts:
                cand = d / f"{stem}{ext}"
                if cand.exists():
                    return cand.resolve()
        return None

    out = {}
    for mid in range(model.nmesh):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        if name is None:
            continue
        p = resolve(name)
        if p is None:
            logger.warning("could not resolve mesh file for %r", name)
        out[mid] = (name, str(p) if p else None)
    return out


# ---------------------------------------------------------------------------
# Scene graph: walk the compiled mjModel into a Blender-friendly JSON dict.
# ---------------------------------------------------------------------------

def build_scene_graph(model, mesh_files: dict | None = None) -> dict:
    """``mesh_files`` ({mesh_id: (name, abs path)}) overrides the PegClimb asset
    search, so a model from another robot stack can reuse this walk."""
    import mujoco

    if mesh_files is None:
        mesh_files = _build_mesh_file_map(model)

    type_to_str = {
        getattr(mujoco.mjtGeom, n): n.replace("mjGEOM_", "").lower()
        for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")
    }
    skip_groups = {3, 4, 5}  # collision / debug layers

    bodies = []
    for bi in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or f"body_{bi}"
        parent = int(model.body_parentid[bi])
        bodies.append({
            "index": bi,
            "name": bname,
            "parent_index": parent,
            "parent_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world",
            "geoms": [],
        })

    n_emit = n_skip = 0
    for gi in range(model.ngeom):
        group = int(model.geom_group[gi])
        if group in skip_groups:
            n_skip += 1
            continue
        gtype = type_to_str.get(int(model.geom_type[gi]), str(int(model.geom_type[gi])))
        bi = int(model.geom_bodyid[gi])
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom_{gi}"

        # Effective colour: material rgba when a material is bound, else the
        # geom's own rgba. (xArm uses materials white/gray; LEAP uses black.)
        matid = int(model.geom_matid[gi])
        mat_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
                    if matid >= 0 else None)
        if matid >= 0:
            rgba = [float(x) for x in model.mat_rgba[matid]]
        else:
            rgba = [float(x) for x in model.geom_rgba[gi]]

        # Drop fully-transparent markers (target site boxes / tip targets).
        if rgba[3] <= 0.0:
            n_skip += 1
            continue

        mesh_id = int(model.geom_dataid[gi])
        mesh_name = mesh_file = None
        mesh_quat_int = (1.0, 0.0, 0.0, 0.0)
        mesh_pos_int = (0.0, 0.0, 0.0)
        if gtype == "mesh" and mesh_id >= 0:
            mesh_name, mesh_file = mesh_files.get(mesh_id, (None, None))
            mesh_quat_int = tuple(float(x) for x in model.mesh_quat[mesh_id])
            mesh_pos_int = tuple(float(x) for x in model.mesh_pos[mesh_id])

        # Strip MuJoCo's internal mesh alignment from the compiled geom pose so
        # the raw (unaligned) Blender-imported mesh lands correctly. See
        # render/replay.py:build_scene_graph for the full derivation.
        geom_pos = tuple(float(x) for x in model.geom_pos[gi])
        geom_quat = tuple(float(x) for x in model.geom_quat[gi])
        if gtype == "mesh" and mesh_id >= 0:
            user_quat = _quat_mul(geom_quat, _quat_conj(mesh_quat_int))
            offset = _quat_rotate(user_quat, mesh_pos_int)
            user_pos = (geom_pos[0] - float(offset[0]),
                        geom_pos[1] - float(offset[1]),
                        geom_pos[2] - float(offset[2]))
        else:
            user_pos, user_quat = geom_pos, geom_quat

        bodies[bi]["geoms"].append({
            "index": gi,
            "name": gname,
            "type": gtype,
            "size": [float(s) for s in model.geom_size[gi]],
            "local_pos": list(user_pos),
            "local_quat": list(user_quat),
            "rgba_geom": rgba,
            "material": mat_name,
            "mesh_name": mesh_name,
            "mesh_file": mesh_file,
            "mesh_scale": [1.0, 1.0, 1.0],
            "group": group,
        })
        n_emit += 1

    cameras = []
    for ci in range(model.ncam):
        cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, ci)
        bi = int(model.cam_bodyid[ci])
        cameras.append({
            "index": ci,
            "name": cname,
            "body_index": bi,
            "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or "world",
            "local_pos": [float(x) for x in model.cam_pos[ci]],
            "local_quat": [float(x) for x in model.cam_quat[ci]],
            "fovy": float(model.cam_fovy[ci]),
        })

    logger.info("scene graph: %d bodies, %d visual geoms (%d skipped), %d cameras",
                model.nbody, n_emit, n_skip, model.ncam)
    return {"bodies": bodies, "cameras": cameras}


# ---------------------------------------------------------------------------
# Policy rollout
# ---------------------------------------------------------------------------

def _make_env(randomize: bool, max_steps: int):
    from POLICY.peg_climb_env import PegClimbEnv
    return PegClimbEnv(
        randomize=randomize, expose_slip=True, slip_encoding="dense",
        fixed_episode_length=True, max_episode_steps=max_steps,
    )


def _rollout(env, model, peg_body, max_steps, capture):
    """Roll one deterministic episode. Returns (climb, success, frames).

    ``frames`` is a list of (xpos copy, xquat copy, time) when capture=True.
    climb = z0 - min(peg z) since the arm is frozen and the hand walks up the
    peg (peg slides down through the grasp as the gait climbs).
    """
    import mujoco  # noqa: F401
    obs, info = env.reset(seed=env._rollout_seed)
    z0 = float(env.data.xpos[peg_body, 2])
    minz = z0
    success = False
    frames = []
    if capture:
        frames.append((env.data.xpos.copy(), env.data.xquat.copy(), float(env.data.time)))
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        minz = min(minz, float(env.data.xpos[peg_body, 2]))
        success = success or bool(info.get("success", False))
        if capture:
            frames.append((env.data.xpos.copy(), env.data.xquat.copy(), float(env.data.time)))
        if term or trunc:
            break
    return (z0 - minz), success, frames


def main() -> None:
    import mujoco
    from stable_baselines3 import PPO
    # Importing make_env registers POLICY.train_peg.PointNetMLPExtractor so
    # PPO.load can resolve the saved features-extractor class.
    from POLICY.train_peg import make_env  # noqa: F401
    from POLICY.peg_climb_env import PEG_NAME

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 0, 4, 3, 5, 2])
    p.add_argument("--randomize", dest="randomize", action="store_true", default=True)
    p.add_argument("--no-randomize", dest="randomize", action="store_false")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--hold-frames", type=int, default=20,
                   help="repeat the final pose this many frames so the climbed "
                        "pose lingers at the end of the animation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(args.policy), device=args.device)
    logger.info("loaded policy %s", args.policy)

    # Pass 1: score each seed, pick best by (success, climb).
    scores = []
    for s in args.seeds:
        env = _make_env(args.randomize, args.max_steps)
        env._rollout_seed = s
        peg = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, PEG_NAME)
        climb, success, _ = _rollout(env, model, peg, args.max_steps, capture=False)
        logger.info("  seed %d: climb=%.4f success=%s", s, climb, success)
        scores.append((success, climb, s))
        del env
    scores.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_seed = scores[0][2]
    logger.info("best seed = %d (success=%s climb=%.4f)",
                best_seed, scores[0][0], scores[0][1])

    # Pass 2: re-run the best seed, capturing frames + the scene graph from the
    # *same* env so geom structure matches the recorded poses.
    env = _make_env(args.randomize, args.max_steps)
    env._rollout_seed = best_seed
    peg = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, PEG_NAME)
    climb, success, frames = _rollout(env, model, peg, args.max_steps, capture=True)
    logger.info("captured %d frames (climb=%.4f success=%s)", len(frames), climb, success)

    scene = build_scene_graph(env.model)
    body_names = [b["name"] for b in scene["bodies"]]
    NB = env.model.nbody

    body_pos = np.stack([f[0] for f in frames], axis=0).astype(np.float32)   # (T,NB,3)
    body_quat = np.stack([f[1] for f in frames], axis=0).astype(np.float32)  # (T,NB,4)
    timestamps = np.array([f[2] for f in frames], dtype=np.float64)

    # Hold the final pose so the climbed configuration lingers on screen.
    if args.hold_frames > 0:
        dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.05
        tail_pos = np.repeat(body_pos[-1:], args.hold_frames, axis=0)
        tail_quat = np.repeat(body_quat[-1:], args.hold_frames, axis=0)
        tail_ts = timestamps[-1] + dt * np.arange(1, args.hold_frames + 1)
        body_pos = np.concatenate([body_pos, tail_pos], axis=0)
        body_quat = np.concatenate([body_quat, tail_quat], axis=0)
        timestamps = np.concatenate([timestamps, tail_ts], axis=0)

    np.savez_compressed(
        args.out / "traj.npz",
        timestamps=timestamps,
        body_pos=body_pos,
        body_quat=body_quat,
        body_names=np.array(body_names),
        climb=np.float32(climb),
        success=np.bool_(success),
        seed=np.int64(best_seed),
    )
    logger.info("wrote %s  (%d frames, %d bodies)", args.out / "traj.npz",
                body_pos.shape[0], NB)

    (args.out / "scene.json").write_text(json.dumps(scene, indent=2))
    logger.info("wrote %s", args.out / "scene.json")

    cameras_out = {c["name"]: {"static": c} for c in scene["cameras"]}
    (args.out / "cameras.json").write_text(json.dumps(cameras_out, indent=2))
    logger.info("wrote %s", args.out / "cameras.json")


if __name__ == "__main__":
    main()
