"""Quick render of a single frame from an assembled scene.blend.

Used as a sanity-check between build_scene.py and the full animation render.

Example::

    uv run python scripts/render_one_frame.py \\
        --blend outputs/trial_15/scene.blend \\
        --frame 130 \\
        --out outputs/trial_15/preview_f130.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import bpy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blend", required=True, type=Path)
    p.add_argument("--frame", type=int, default=1)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--samples", type=int, default=64)
    args = p.parse_args()

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.frame_set(args.frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(args.out.resolve())
    bpy.ops.render.render(write_still=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
