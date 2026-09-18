"""Render a rollout video for any task exported by ``replay_rebuttal.py``.

The video companion to ``render_rebuttal_strip.py``: same scene build, same
object markers and collision-mesh fixes (``add_markers``), same materials,
lighting and camera framing, so a clip and its figure row show the same thing.
Every exported step from the start of the episode to a little past the first
success (``TAIL``) is rendered in Cycles and encoded to mp4. Episodes with no
success (cube, whose env bar is a full turn) run to the end.

Style defaults to "rebuttal": every manipulated object in the same blue, with
tight framing, so the eight task videos read as one set.

Runs under the ``blender-render`` env; one GPU per process is enough, so give
each task its own ``CUDA_VISIBLE_DEVICES`` to render them in parallel::

    CUDA_VISIBLE_DEVICES=0 conda run -n blender-render python render/render_rebuttal_video.py \\
        --replay outputs/paper_final/cube_rotation --task cube_rotation \\
        --out outputs/video/cube_rotation.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

import bpy

_RENDER_DIR = Path(__file__).resolve().parent
if str(_RENDER_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_RENDER_DIR.parent))
from render import build_scene as bs  # noqa: E402
from render import render_rebuttal_strip as rs  # noqa: E402

logger = logging.getLogger("render_rebuttal_video")


def video_steps(traj: dict, task: str, stride: int, tail: float | None = None) -> list[int]:
    T = traj["body_pos"].shape[0]
    s = int(traj["success_step"])
    tail = rs.TAIL[task] if tail is None else tail
    end = T - 1 if s < 0 else min(T - 1, int(round(s * (1.0 + tail))))
    return list(range(0, end + 1, stride))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True, help="replay_rebuttal.py output dir")
    p.add_argument("--task", required=True, choices=sorted(rs.TAIL))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--style", choices=["paper", "rebuttal"], default="rebuttal")
    p.add_argument("--w", type=int, default=1280)
    p.add_argument("--h", type=int, default=720)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--stride", type=int, default=1, help="render every k-th exported step")
    # Steps are exported at the 66.7 Hz control rate, so 30 fps plays at about
    # 0.45x: slow enough to follow the fingers on the short tasks.
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--view", default=None, metavar="DIST,AZIM,ELEV[,DZ]", help="override the task camera")
    p.add_argument("--max-frames", type=int, default=None, help="render only the first N (framing tests)")
    p.add_argument("--tail", type=float, default=None,
                   help="run this fraction of the success step past success (default: the strip's TAIL). "
                        "Not for cracker_climb: its box is weight-compensated and drifts off after success.")
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    task = args.task
    rs._STYLE[0] = args.style
    if args.style == "rebuttal":
        rs.VIEWS.update(rs.VIEWS_TIGHT)
    if args.view:
        v = [float(x) for x in args.view.split(",")]
        rs.VIEWS[task] = dict(dist=v[0], azim=v[1], elev=v[2], lookat_off=(0.0, 0.0, v[3] if len(v) > 3 else 0.0))

    scene = json.loads((args.replay / "scene.json").read_text())
    traj = dict(np.load(args.replay / "traj.npz", allow_pickle=True))
    steps = video_steps(traj, task, args.stride, args.tail)
    if args.max_frames:
        steps = steps[:args.max_frames]
    logger.info("%s: %d exported steps, success_step=%d, rendering %d frames",
                task, traj["body_pos"].shape[0], int(traj["success_step"]), len(steps))

    bs.reset_scene()
    if hasattr(bs, "_reset_material_cache"):
        bs._reset_material_cache()
    rs.add_markers(scene, task)
    rs._CURRENT_TASK[0] = task
    empties = rs.build(scene)
    names = [str(x) for x in traj["body_names"]]
    obj_i = names.index("peg")
    # A fixed camera on the object's mean position over the clip, as for the strip.
    v = rs.VIEWS[task]
    lookat = traj["body_pos"][steps, obj_i].mean(axis=0) + np.array(v.get("lookat_off", (0, 0, 0)))
    rs.add_free_camera(lookat.tolist(), v["dist"], v["azim"], v["elev"])
    bs.setup_world_lighting(strength=0.45)
    bs.add_mjcf_style_lights()
    bs.configure_cycles(samples=args.samples, res_x=args.w, res_y=args.h)
    bpy.context.scene.view_settings.exposure = -0.35
    rs.enable_gpu(args.samples)

    frames_dir = args.out.with_suffix("") / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    scn = bpy.context.scene
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    for k, t in enumerate(steps):
        path = frames_dir / f"{k:05d}.png"
        if path.exists():  # resumable: a killed render picks up where it stopped
            continue
        rs.pose(empties, traj, t)
        bpy.context.view_layer.update()
        scn.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        if k % 25 == 0:
            logger.info("  %s frame %d/%d (step %d)", task, k + 1, len(steps), t)

    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(args.fps),
           "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-movflags", "+faststart", str(args.out)]
    subprocess.run(cmd, check=True)
    logger.info("wrote %s (%d frames, %.1f s at %d fps)", args.out, len(steps), len(steps) / args.fps, args.fps)


if __name__ == "__main__":
    main()
