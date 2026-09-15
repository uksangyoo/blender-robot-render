"""Real-world rollout figure: one row per task, N wide 16:9 frames, black ground.

Frames are evenly spaced over each video's task portion and cropped to a 16:9
window around hand and object. No labels, thin black gutters -- the style of
the lab's existing real-robot rollout strips.

    conda run -n blender-render python render/real_rollout_figure.py \\
        --video-dir ~/Projects/vibereact/POLICY/figures \\
        --out ~/Projects/vibereact/POLICY/figures/real_world_rollouts.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image

# (video, first frame, last frame, 16:9 crop box x0,y0,x1,y1 in source pixels).
# The tube video's last ~20 frames set the can down after the climb.
ROWS = [
    ("cracker_task.mp4", 0, 252, (0, 90, 1600, 990)),          # Box Climb
    ("tube_task.mp4", 0, 300, (320, 0, 1920, 900)),             # Can Climb
    ("hex_task (1) (2).mp4", 0, 821, (140, 0, 1280, 641)),     # Nut Rotation
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tile-w", type=int, default=960)
    ap.add_argument("--col-gap", type=int, default=24)
    ap.add_argument("--row-gap", type=int, default=44)
    ap.add_argument("--margin-x", type=int, default=100)
    ap.add_argument("--margin-y", type=int, default=30)
    args = ap.parse_args()

    tw, th = args.tile_w, round(args.tile_w * 9 / 16)
    W = 2 * args.margin_x + args.n * tw + (args.n - 1) * args.col_gap
    H = 2 * args.margin_y + len(ROWS) * th + (len(ROWS) - 1) * args.row_gap
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    for r, (name, a, b, box) in enumerate(ROWS):
        frames = [f for f in iio.get_reader(str(args.video_dir / name))]
        idx = np.linspace(a, min(b, len(frames) - 1), args.n).round().astype(int)
        y = args.margin_y + r * (th + args.row_gap)
        for k, i in enumerate(idx):
            tile = Image.fromarray(frames[i]).crop(box).resize((tw, th), Image.LANCZOS)
            canvas.paste(tile, (args.margin_x + k * (tw + args.col_gap), y))
        print(f"{name}: frames {idx.tolist()}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    canvas.save(args.out.with_suffix(".pdf"), resolution=300.0)
    print(f"wrote {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
