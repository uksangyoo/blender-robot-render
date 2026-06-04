"""Render a built PegClimb scene.blend to frames + an mp4.

Unlike scripts/render_animation.py this re-applies the Cycles GPU/OPTIX
device preferences before rendering — those live in the Blender *user*
preferences (not the .blend), so a fresh ``bpy`` process otherwise falls
back to CPU. Enables every non-CPU device, so both RTX 4090s are used.

    conda run -n blender-render python \\
        thirdparty/blender-robot-render/render/render_pegclimb.py \\
        --blend  .../peg_climb_seed3/scene.blend \\
        --out-dir .../peg_climb_seed3/frames \\
        --video  .../peg_climb_seed3/peg_climb.mp4 \\
        --samples 128 --fps 30
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import bpy

logger = logging.getLogger("render_pegclimb")


def enable_gpu(samples: int) -> None:
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = "OPTIX"
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.adaptive_threshold = 0.01
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    scn.cycles.device = "GPU"
    used = []
    for d in prefs.devices:
        d.use = (d.type != "CPU")
        if d.use:
            used.append(d.name)
    logger.info("Cycles OPTIX devices enabled: %s", used or "<none — CPU fallback!>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blend", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    enable_gpu(args.samples)

    scn = bpy.context.scene
    if args.start is not None:
        scn.frame_start = args.start
    if args.end is not None:
        scn.frame_end = args.end
    scn.frame_step = args.step
    scn.render.fps = args.fps

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.filepath = str(args.out_dir.resolve()) + "/frame_"

    logger.info("rendering frames [%d, %d] step=%d samples=%d -> %s",
                scn.frame_start, scn.frame_end, scn.frame_step, args.samples, args.out_dir)
    bpy.ops.render.render(animation=True)

    if args.video is not None:
        import imageio
        frame_files = sorted(args.out_dir.glob("frame_*.png"))
        if not frame_files:
            logger.warning("no frames in %s", args.out_dir)
            return
        args.video.parent.mkdir(parents=True, exist_ok=True)
        logger.info("encoding %d frames -> %s @ %d fps", len(frame_files), args.video, args.fps)
        with imageio.get_writer(str(args.video), fps=args.fps, codec="libx264",
                                quality=8, pixelformat="yuv420p") as w:
            for f in frame_files:
                w.append_data(imageio.v3.imread(str(f)))
        logger.info("wrote %s", args.video)


if __name__ == "__main__":
    main()
