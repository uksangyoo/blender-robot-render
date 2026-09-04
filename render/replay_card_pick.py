"""Export a card_pick policy rollout for Blender rendering.

Same ``scene.json`` / ``cameras.json`` / ``traj.npz`` triple as
``replay_peg_climb.py``, but rolls out the GPU (MuJoCo Warp) env and one of the
batched PointNet policies, so it must run in the ``vibewarp`` env:

    conda run -n vibewarp python thirdparty/blender-robot-render/render/replay_card_pick.py \\
        --policy output/policies/CP6_tactile_s0/final_model.pt \\
        --out thirdparty/blender-robot-render/outputs/card_tactile --pick success

`--pick success` selects a world that completed the pick (the tactile arm);
`--pick failure` selects a representative world that did not, preferring one
that never managed two fingertips on the card — which is what the no-tactile
policy actually does.

Actions are the policy MEAN, not a sample: `PointNetActorCritic.act()` draws
from the Gaussian, which is right for training but makes a render jittery.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_peg_climb import build_scene_graph  # noqa: E402

logger = logging.getLogger("replay_card_pick")


def main() -> None:
    from POLICY.gpu.card_pick_env import WarpCardPickEnv
    from POLICY.gpu.ppo import PointNetActorCritic, PPOConfig

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=700)
    p.add_argument("--stride", type=int, default=2,
                   help="keep every Nth control step as an animation frame")
    p.add_argument("--pick", choices=("success", "failure", "maxsep"), default="success")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hold-frames", type=int, default=20)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()

    logging.basicConfig(level=getattr(logging, a.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(a.device)

    ck = torch.load(a.policy, map_location=a.device, weights_only=False)
    obs_mode = ck.get("config", {}).get("obs_mode", "full")
    env = WarpCardPickEnv(num_envs=a.num_envs, seed=a.seed, device=a.device,
                          randomize=True, max_episode_steps=a.max_steps,
                          expose_slip=obs_mode == "full")
    cfg = PPOConfig(use_proprio=obs_mode in ("full", "pc_proprio"),
                    use_slip=obs_mode == "full", compile=False)
    policy = PointNetActorCritic(env.observation_shapes, env.action_dim, cfg).to(a.device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    logger.info("loaded %s (obs_mode=%s)", a.policy, obs_mode)

    import warp as wp
    N = a.num_envs
    obs = env.reset()
    xp = wp.to_torch(env.d.xpos)
    NB = xp.shape[1]
    frames_pos, frames_quat, ts = [], [], []
    two_tip = torch.zeros(N, device=a.device)
    touch = torch.zeros(N, device=a.device)
    best_err = torch.full((N,), 1e9, device=a.device)
    # one control step of sim time
    from POLICY.gpu.cube_env import RL_DT

    with torch.no_grad():
        for t in range(a.max_steps - 10):   # stop before the auto-reset wipes flags
            mean, _ = policy(obs)
            obs, r, d, info = env.step(mean)
            nt = env._n_tips_on_card()
            two_tip += (nt >= 2).float()
            touch += (nt >= 1).float()
            best_err = torch.minimum(best_err,
                                     (env._separation() - env._target_sep).abs())
            if t % a.stride == 0:
                frames_pos.append(wp.to_torch(env.d.xpos).clone().cpu().numpy())
                frames_quat.append(wp.to_torch(env.d.xquat).clone().cpu().numpy())
                ts.append(t * RL_DT)

    succ = env._success.clone()
    n_steps = a.max_steps - 10
    two_frac = (two_tip / n_steps).cpu().numpy()
    touch_frac = (touch / n_steps).cpu().numpy()
    err_mm = (best_err * 1000).cpu().numpy()
    logger.info("rollout: %d/%d worlds succeeded; 2tip mean %.2f; touch mean %.2f",
                int(succ.sum()), N, two_frac.mean(), touch_frac.mean())

    ok = succ.cpu().numpy()
    final_sep = (env._separation() * 1000).cpu().numpy()
    if a.pick == "maxsep":
        # The most LEGIBLE success: still a completed pick, but the card ends as
        # far off the stack as the task allows, so the overhang reads on camera.
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            raise SystemExit("no successful world; try another seed")
        w = int(cand[np.argmax(final_sep[cand])])
        logger.info("maxsep: %d successes, final separations %s",
                    cand.size, np.round(np.sort(final_sep[cand])[::-1][:8], 1))
    elif a.pick == "success":
        cand = np.flatnonzero(ok)
        if cand.size == 0:
            raise SystemExit("no successful world in this rollout — raise --num-envs "
                             "or try another seed")
        # among successes, the one that held two fingertips longest reads best
        w = int(cand[np.argmax(two_frac[cand])])
    else:
        cand = np.flatnonzero(~ok)
        # the representative failure: least contact, i.e. what the no-tactile
        # policy actually does — reach the vicinity without ever holding the card
        w = int(cand[np.argmin(two_frac[cand])])
    logger.info("picked world %d: success=%s  2tip=%.2f  touch=%.2f  best_err=%.2f mm "
                "final_sep=%.1f mm", w, bool(ok[w]), two_frac[w], touch_frac[w],
                err_mm[w], final_sep[w])

    body_pos = np.stack([f[w] for f in frames_pos]).astype(np.float32)
    body_quat = np.stack([f[w] for f in frames_quat]).astype(np.float32)
    timestamps = np.asarray(ts, dtype=np.float64)

    if a.hold_frames > 0:
        dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.05
        body_pos = np.concatenate([body_pos, np.repeat(body_pos[-1:], a.hold_frames, 0)])
        body_quat = np.concatenate([body_quat, np.repeat(body_quat[-1:], a.hold_frames, 0)])
        timestamps = np.concatenate(
            [timestamps, timestamps[-1] + dt * np.arange(1, a.hold_frames + 1)])

    scene = build_scene_graph(env.mjm)
    body_names = [b["name"] for b in scene["bodies"]]
    assert len(body_names) == NB, f"{len(body_names)} scene bodies vs {NB} in data"

    np.savez_compressed(
        a.out / "traj.npz", timestamps=timestamps, body_pos=body_pos,
        body_quat=body_quat, body_names=np.array(body_names),
        success=np.bool_(ok[w]), two_tip_frac=np.float32(two_frac[w]),
        touch_frac=np.float32(touch_frac[w]), best_err_mm=np.float32(err_mm[w]),
        world=np.int64(w), obs_mode=np.array(obs_mode))
    (a.out / "scene.json").write_text(json.dumps(scene, indent=2))
    (a.out / "cameras.json").write_text(
        json.dumps({c["name"]: {"static": c} for c in scene["cameras"]}, indent=2))
    logger.info("wrote %s (%d frames, %d bodies)", a.out, body_pos.shape[0], NB)


if __name__ == "__main__":
    main()
