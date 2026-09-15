"""Compose sim progression rows with real-world rollout frames.

Sim tiles come from ``render_rebuttal_strip.py`` (``<tiles_dir>/<task>_<k>.png``).
For each task with a real video, 12 frames are taken evenly over the task
portion of the video, square-cropped around hand and object, resized to the
sim tile size, and placed in a real-world block under all the sim rows.

    conda run -n blender-render python render/compose_sim_real.py \\
        --tiles-dir outputs/paper_rstyle/tiles --video-dir ~/Projects/vibereact/POLICY/figures \\
        --out outputs/paper_rstyle/paper_task_progression_sim_real.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS = [
    ("cracker_climb", "Box Climb"),
    ("peg_climb", "Can Climb"),
    ("peg_in_hole", "Peg in Hole"),
    ("cube_rotation", "Cube Rotation"),
    ("hex_nut_fingers", "Nut Rotation"),
]
# task -> (video file, crop x0, y0, side in source pixels, first frame, last frame).
# Frame ranges cover the task: the tube video's last ~20 frames set the can
# down after the climb, so its strip ends before that.
REAL = {
    "cracker_climb": ("cracker_task.mp4", 180, 60, 1020, 0, 252),
    "peg_climb": ("tube_task.mp4", 560, 0, 1060, 0, 300),
    "hex_nut_fingers": ("hex_task (1) (2).mp4", 330, 0, 720, 0, 821),
}


def font(size: int, bold: bool):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def real_tiles(video: Path, x0, y0, side, a, b, n, tile) -> list[Image.Image]:
    frames = [fr for fr in iio.get_reader(str(video))]
    idx = np.linspace(a, min(b, len(frames) - 1), n).round().astype(int)
    return [Image.fromarray(frames[i]).crop((x0, y0, x0 + side, y0 + side)).resize((tile, tile), Image.LANCZOS)
            for i in idx]


def label_panel(h: int, w: int, title: str, sub: str) -> Image.Image:
    """Two-line label (bold task, regular Sim/Real) rotated to run up the row."""
    tmp = Image.new("RGB", (h, w), (255, 255, 255))
    d = ImageDraw.Draw(tmp)
    ft, fs = font(max(18, w // 3), True), font(max(16, int(w / 3.6)), False)
    d.text((h // 2, int(w * 0.36)), title, fill=(20, 20, 20), font=ft, anchor="mm")
    d.text((h // 2, int(w * 0.76)), sub, fill=(90, 90, 90), font=fs, anchor="mm")
    return tmp.rotate(90, expand=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--group-gap", type=int, default=34)
    ap.add_argument("--label-w", type=int, default=150)
    args = ap.parse_args()

    # All simulation rows first, then the real-world rollouts together at the
    # bottom (same task order). The flag marks rows that start a new block.
    rows = []  # (title, sub, tiles, starts_block)
    for task, title in ROWS:
        sim = [Image.open(p).convert("RGB") for p in sorted(args.tiles_dir.glob(f"{task}_[0-9][0-9].png"))]
        if len(sim) != args.n:
            raise SystemExit(f"{task}: expected {args.n} sim tiles in {args.tiles_dir}, found {len(sim)}")
        rows.append((title, "Sim", sim, False))
    tile_px = rows[0][2][0].width
    first_real = True
    for task, title in ROWS:
        if task in REAL:
            f, x0, y0, side, a, b = REAL[task]
            rows.append((title, "Real", real_tiles(args.video_dir / f, x0, y0, side, a, b, args.n, tile_px), first_real))
            first_real = False

    tile = rows[0][2][0].width
    W = args.label_w + args.n * tile + (args.n - 1) * args.gap
    ys, y = [], 0
    for i, r in enumerate(rows):
        if i > 0:
            y += args.group_gap if r[3] else args.gap  # sim -> real block: larger gap
        ys.append(y)
        y += tile
    canvas = Image.new("RGB", (W, y), (255, 255, 255))
    for (title, sub, tiles, _), yy in zip(rows, ys):
        for k, t in enumerate(tiles):
            canvas.paste(t, (args.label_w + k * (tile + args.gap), yy))
        canvas.paste(label_panel(tile, args.label_w, title, sub), (0, yy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    canvas.save(args.out.with_suffix(".pdf"), resolution=300.0)
    print(f"wrote {args.out} ({W}x{y}), rows: {[(r[0], r[1]) for r in rows]}")


if __name__ == "__main__":
    main()
