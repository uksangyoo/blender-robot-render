"""Render the full animated scene to a frames directory + assemble to mp4.

Example::

    uv run python scripts/render_animation.py \\
        --blend outputs/trial_15/scene.blend \\
        --out-dir outputs/trial_15/frames \\
        --samples 64 \\
        --fps 24 \\
        --video outputs/trial_15/render.mp4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import bpy

logger = logging.getLogger("render_animation")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blend", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--res-x", type=int, default=None,
                   help="Override render resolution width (defaults to .blend value)")
    p.add_argument("--res-y", type=int, default=None,
                   help="Override render resolution height (defaults to .blend value)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    scn = bpy.context.scene
    scn.cycles.samples = args.samples
    if args.start is not None:
        scn.frame_start = args.start
    if args.end is not None:
        scn.frame_end = args.end
    scn.frame_step = args.step
    scn.render.fps = args.fps
    if args.res_x is not None:
        scn.render.resolution_x = args.res_x
    if args.res_y is not None:
        scn.render.resolution_y = args.res_y

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.compression = 50
    # Render writes to <filepath>####.png with the frame number suffix.
    scn.render.filepath = str(args.out_dir.resolve()) + "/frame_"

    logger.info(
        "rendering frames [%d, %d] step=%d samples=%d → %s",
        scn.frame_start, scn.frame_end, scn.frame_step, args.samples, args.out_dir,
    )
    bpy.ops.render.render(animation=True)

    if args.video is not None:
        # Encode with ffmpeg via imageio (already in deps).
        import imageio
        frame_files = sorted(args.out_dir.glob("frame_*.png"))
        if not frame_files:
            logger.warning("no frames found in %s", args.out_dir)
            return
        logger.info("encoding %d frames → %s @ %d fps", len(frame_files), args.video, args.fps)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(args.video), fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p"
        ) as writer:
            for f in frame_files:
                writer.append_data(imageio.v3.imread(str(f)))
        logger.info("wrote %s (%d frames)", args.video, len(frame_files))


if __name__ == "__main__":
    main()
