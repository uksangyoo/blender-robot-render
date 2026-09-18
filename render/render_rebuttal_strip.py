"""Render a task-progression figure for the CoRL-rebuttal tasks.

For each task exported by ``replay_rebuttal.py`` this builds the Blender scene
(reusing the peg-climb body/geom pass and materials), poses the bodies directly
at N frames spaced from the start of the episode to a little past the first
success, and renders one square Cycles tile per frame. The tiles are then
composed into one figure: a row per task, progression left to right.

Runs under the ``blender-render`` env (Python 3.11 + bpy)::

    conda run -n blender-render python render/render_rebuttal_strip.py \\
        --replay-root outputs/rebuttal --out outputs/rebuttal/progression.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

import bpy
from mathutils import Vector

_RENDER_DIR = Path(__file__).resolve().parent
if str(_RENDER_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_RENDER_DIR.parent))
from render import build_scene as bs  # noqa: E402
from render import build_scene_pegclimb as pc  # noqa: E402

logger = logging.getLogger("render_rebuttal_strip")

TASKS = [
    ("phone_flip", "Phone Flip"),
    ("screwdriver_turn", "Screwdriver"),
    ("card_pickup", "Card Pickup"),
    # Paper tasks (Fig. 6), rendered with each object's own colours.
    ("cracker_climb", "Box Climb"),
    ("peg_climb", "Can Climb"),
    ("peg_in_hole", "Peg in Hole"),
    ("cube_rotation", "Cube Rotation"),
    ("hex_nut_fingers", "Nut Rotation"),
]
PAPER_TASKS = {"cracker_climb", "peg_climb", "peg_in_hole", "cube_rotation", "hex_nut_fingers"}
_CURRENT_TASK = [""]
# "paper": paper tasks keep each object's Fig. 6 colour. "rebuttal": every task
# is drawn like the rebuttal figure -- blue held object, wood/metal props,
# tight framing on hand and object -- so the two figures read as one set.
_STYLE = ["paper"]

# Free-camera framing per task (MuJoCo convention: azimuth/elevation in degrees,
# camera sits `dist` back from `lookat` along the view direction). `lookat_off`
# is added to the object's mean position over the rendered frames.
VIEWS = {
    # Side-on along the slab's roll axis: the dark screen turns away as it flips.
    "phone_flip": dict(dist=0.38, azim=90.0, elev=-30.0, lookat_off=(0.0, 0.0, 0.0)),
    # The stripe on the handle swings round as the tool turns in the socket.
    "screwdriver_turn": dict(dist=0.41, azim=210.0, elev=-28.0, lookat_off=(0.0, 0.0, 0.0)),
    # Card slides to the table edge, then comes up into the grasp.
    "card_pickup": dict(dist=0.42, azim=140.0, elev=-22.0, lookat_off=(0.0, 0.0, 0.0)),
    "cracker_climb": dict(dist=0.68, azim=150.0, elev=-16.0, lookat_off=(0.0, 0.0, 0.02)),
    "peg_climb": dict(dist=0.56, azim=150.0, elev=-16.0, lookat_off=(0.0, 0.0, 0.0)),
    "peg_in_hole": dict(dist=0.66, azim=150.0, elev=-22.0, lookat_off=(0.0, 0.0, 0.0)),
    "cube_rotation": dict(dist=0.40, azim=150.0, elev=-30.0, lookat_off=(0.0, 0.0, 0.0)),
    "hex_nut_fingers": dict(dist=0.52, azim=150.0, elev=-22.0, lookat_off=(0.0, 0.0, 0.03)),
}
# Tighter framing for the paper tasks in the rebuttal style.
VIEWS_TIGHT = {
    "cracker_climb": dict(dist=0.62, azim=150.0, elev=-14.0, lookat_off=(0.0, 0.0, 0.035)),
    "peg_climb": dict(dist=0.52, azim=150.0, elev=-14.0, lookat_off=(0.0, 0.0, 0.03)),
    "peg_in_hole": dict(dist=0.52, azim=150.0, elev=-24.0, lookat_off=(0.0, 0.0, 0.0)),
    "cube_rotation": dict(dist=0.35, azim=150.0, elev=-34.0, lookat_off=(0.0, 0.0, 0.0)),
    "hex_nut_fingers": dict(dist=0.40, azim=150.0, elev=-8.0, lookat_off=(0.0, 0.0, 0.015)),
}
# How far past the first success the strip runs, as a fraction of the success step.
TAIL = {"phone_flip": 0.35, "screwdriver_turn": 0.30, "card_pickup": 0.6,
        # Box Climb's box is weight-compensated: past success it drifts off the
        # hand, so its row ends at success.
        "cracker_climb": 0.0, "peg_climb": 0.2, "peg_in_hole": 0.3, "cube_rotation": 0.2,
        "hex_nut_fingers": 0.2}

OBJECT_BLUE = (0.07, 0.30, 0.74, 1.0)
SCREEN_BLACK = (0.015, 0.016, 0.02, 1.0)
BOLT_RAISE = 0.025  # m, cosmetic: see add_markers (hex_nut_fingers)
STRIPE_WHITE = (0.92, 0.92, 0.90, 1.0)
_PEGCLIMB_MATERIAL = pc.material_for_geom


def _mat(name, color, rough=0.4, metal=0.0, bump=0.1, scale=300.0):
    return bs.make_principled_material(name, base_color=color, roughness=rough, metallic=metal,
                                       specular=0.5, noise_bump=bump, noise_scale=scale,
                                       roughness_variation=0.08)


def material_for_geom(body_name: str, geom: dict):
    b = (body_name or "").lower()
    g = (geom.get("name") or "").lower()
    if g == "marker_screen":
        return _mat("mat__screen", SCREEN_BLACK, rough=0.12, bump=0.0)
    if g == "marker_stripe":
        return _mat("mat__stripe", STRIPE_WHITE, rough=0.4, bump=0.0)
    if _STYLE[0] == "rebuttal":
        if b == "hole_base":
            return _mat("mat__workpiece_wood", (0.52, 0.38, 0.24, 1.0), rough=0.6, bump=0.2)
        if b == "screw":
            return _mat("mat__socket_metal", (0.62, 0.63, 0.66, 1.0), rough=0.3, metal=0.8)
    if _STYLE[0] == "paper" and _CURRENT_TASK[0] in PAPER_TASKS and b not in ("world",) and not b.startswith(("leap", "link")) \
            and geom.get("type") != "plane":
        rgba = geom.get("rgba_geom") or [0.7, 0.7, 0.7, 1.0]
        key = "_".join(f"{x:.2f}" for x in rgba[:3])
        return _mat(f"mat__obj_{key}", tuple(rgba[:3]) + (1.0,), rough=0.38, bump=0.05)
    if b == "peg":
        return _mat("mat__object_blue", OBJECT_BLUE, rough=0.32, bump=0.05)
    if g.startswith("socket"):
        return _mat("mat__socket_metal", (0.62, 0.63, 0.66, 1.0), rough=0.3, metal=0.8)
    if b == "workpiece":
        return _mat("mat__workpiece_wood", (0.52, 0.38, 0.24, 1.0), rough=0.6, bump=0.2)
    if b == "card_table":
        return _mat("mat__card_table", (0.80, 0.76, 0.70, 1.0), rough=0.55, bump=0.12)
    return _PEGCLIMB_MATERIAL(body_name, geom)


def add_markers(scene: dict, task: str) -> None:
    """Swap in / add object geometry that makes the motion readable in a still.

    Card: the collision box replaces the visual mesh (the two disagree in size,
    and the box is what the policy actually picks up). Phone: a dark screen on
    the slab's +z face, so a flip reads as the screen turning over. Screwdriver:
    a stripe down the handle, so rotation about the axis is visible.
    """
    body = next(b for b in scene["bodies"] if b["name"] == "peg")
    ident = [1.0, 0.0, 0.0, 0.0]
    if task == "cube_rotation":
        # The cube is 4-fold symmetric about its turning axis; an off-centre
        # square on the top face and a stripe on one side make a turn visible.
        h = 0.03
        body["geoms"].append(dict(name="marker_screen", type="box", size=[0.007, 0.007, 0.0005],
                                  local_pos=[0.014, 0.014, h + 0.0004], local_quat=ident))
        body["geoms"].append(dict(name="marker_screen", type="box", size=[0.0005, 0.004, 0.022],
                                  local_pos=[h + 0.0004, 0.0, 0.0], local_quat=ident))
        return
    if task == "cracker_climb":
        # The env's cracker visual mesh sits ~10 cm above its collision box
        # (visual geom centre (-0.015, -0.014, +0.059) vs box (0, 0, -0.043) in
        # the body frame); the box the fingers actually hold is the collision
        # one. Draw that.
        box = {c["name"]: c for c in scene.get("object_collision", [])}.get("peg_geom")
        if box is not None:
            vis = next((g for g in body["geoms"] if g["name"] == "cracker_visual_geom"), {})
            body["geoms"] = [g for g in body["geoms"] if g["name"] != "cracker_visual_geom"]
            body["geoms"].append(dict(name="cracker_box", type="box", size=box["size"],
                                      local_pos=box["local_pos"], local_quat=box["local_quat"],
                                      rgba_geom=vis.get("rgba_geom", [0.5, 0.5, 0.5, 1.0])))
        return
    if task == "hex_nut_fingers":
        # The nut rides a slide+hinge joint that stands in for the threads, and
        # at reset its underside is 13 mm above the bolt's top (the bolt has no
        # collision). Draw the bolt 25 mm taller so its tip sits inside the nut
        # bore, where the threads would engage; the nut itself is not moved.
        screw = next((b for b in scene["bodies"] if b["name"] == "screw"), None)
        if screw is not None:
            for g in screw["geoms"]:
                g["local_pos"] = [g["local_pos"][0], g["local_pos"][1], g["local_pos"][2] + BOLT_RAISE]
        # A radial stripe across the nut's top face (6-fold symmetric otherwise).
        body["geoms"].append(dict(name="marker_screen", type="box", size=[0.0095, 0.0022, 0.0006],
                                  local_pos=[0.0185, 0.0, 0.0126], local_quat=ident))
        # ...and a vertical stripe on one side flat, since the fingers hide the top
        # face from the near-horizontal nut camera. The +x face is a flat (apothem
        # 28 mm; corners at +/-30, 90, 150 deg) and the nut is 25 mm thick.
        body["geoms"].append(dict(name="marker_screen", type="box", size=[0.0005, 0.0035, 0.0105],
                                  local_pos=[0.0284, 0.0, 0.0], local_quat=ident))
        return
    if task in PAPER_TASKS:
        return
    col = {c["name"]: c for c in scene.get("object_collision", [])}
    box = col.get("peg_geom")
    if box is None:
        return
    ident = [1.0, 0.0, 0.0, 0.0]
    if task == "card_pickup":
        body["geoms"] = [g for g in body["geoms"] if g["name"] != "object_visual_geom"]
        body["geoms"].append(dict(name="card_box", type="box", size=box["size"],
                                  local_pos=box["local_pos"], local_quat=box["local_quat"]))
    elif task == "phone_flip":
        sx, sy, sz = box["size"]
        body["geoms"].append(dict(name="marker_screen", type="box",
                                  size=[sx * 0.93, sy * 0.95, 0.0004],
                                  local_pos=[0.0, 0.0, sz + 0.0003], local_quat=ident))
    elif task == "screwdriver_turn":
        r, h = box["size"][0], box["size"][1]
        body["geoms"].append(dict(name="marker_stripe", type="box",
                                  size=[0.0018, 0.0050, h * 0.85],
                                  local_pos=[r - 0.0008, 0.0, box["local_pos"][2]], local_quat=ident))


def build(scene: dict) -> dict:
    pc.material_for_geom = material_for_geom  # build_bodies_and_geoms looks it up by name
    try:
        return pc.build_bodies_and_geoms(scene, floor_expand=6.0)
    finally:
        pc.material_for_geom = _PEGCLIMB_MATERIAL


def pose(body_empties: dict, traj: dict, t: int) -> None:
    for bi, name in enumerate(traj["body_names"]):
        obj = body_empties.get(str(name))
        if obj is None:
            continue
        obj.location = Vector([float(x) for x in traj["body_pos"][t, bi]])
        obj.rotation_quaternion = bs.quat_wxyz(traj["body_quat"][t, bi])


def add_free_camera(lookat, dist, azim, elev, fov_deg=40.0, res=800):
    a, e = math.radians(azim), math.radians(elev)
    fwd = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    loc = Vector(lookat) - dist * fwd
    cam_data = bpy.data.cameras.new("strip_cam")
    cam = bpy.data.objects.new("cam__strip", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = loc
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (Vector(lookat) - loc).to_track_quat("-Z", "Y")
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(fov_deg)
    cam_data.clip_start = 0.01
    scn = bpy.context.scene
    scn.camera = cam
    scn.render.resolution_x = res
    scn.render.resolution_y = res
    return cam


def enable_gpu(samples: int) -> None:
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.samples = samples
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.adaptive_threshold = 0.02
    scn.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scn.cycles.device = "GPU"
                scn.cycles.denoiser = "OPTIX" if backend == "OPTIX" else "OPENIMAGEDENOISE"
                logger.info("Cycles %s on %d GPU(s)", backend, len(gpus))
                return
        except TypeError:
            continue
    scn.cycles.device = "CPU"
    scn.cycles.denoiser = "OPENIMAGEDENOISE"
    logger.warning("no GPU backend; rendering on CPU")


def frame_indices(traj: dict, task: str, n: int) -> list[int]:
    T = traj["body_pos"].shape[0]
    s = int(traj["success_step"])
    end = T - 1 if s < 0 else min(T - 1, int(round(s * (1.0 + TAIL[task]))))
    return [int(round(x)) for x in np.linspace(0, end, n)]


def render_task(task: str, rd: Path, n: int, samples: int, res: int, tiles_dir: Path,
                only: list[int] | None = None, tag: str = "") -> list[Path]:
    scene = json.loads((rd / "scene.json").read_text())
    traj = dict(np.load(rd / "traj.npz", allow_pickle=True))
    idx = frame_indices(traj, task, n)
    keep = list(range(len(idx))) if only is None else [k for k in only if k < len(idx)]
    logger.info("%s: %d frames, success_step=%d, rendering steps %s",
                task, traj["body_pos"].shape[0], int(traj["success_step"]), idx)

    bs.reset_scene()
    if hasattr(bs, "_reset_material_cache"):
        bs._reset_material_cache()
    add_markers(scene, task)
    _CURRENT_TASK[0] = task
    empties = build(scene)
    names = [str(x) for x in traj["body_names"]]
    obj_i = names.index("peg")
    lookat = traj["body_pos"][idx, obj_i].mean(axis=0) + np.array(VIEWS[task].get("lookat_off", (0, 0, 0)))
    v = VIEWS[task]
    add_free_camera(lookat.tolist(), v["dist"], v["azim"], v["elev"], res=res)
    bs.setup_world_lighting(strength=0.45)
    bs.add_mjcf_style_lights()
    bs.configure_cycles(samples=samples, res_x=res, res_y=res)
    bpy.context.scene.view_settings.exposure = -0.35
    enable_gpu(samples)

    out = []
    tiles_dir.mkdir(parents=True, exist_ok=True)
    scn = bpy.context.scene
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    for k in keep:
        t = idx[k]
        pose(empties, traj, t)
        bpy.context.view_layer.update()
        path = tiles_dir / f"{task}{tag}_{k:02d}.png"
        scn.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        out.append(path)
        logger.info("  %s tile %d/%d (step %d)", task, k + 1, n, t)
    return out


def compose(rows: list[tuple[str, list[Path]]], out: Path, gap: int, label_w: int) -> None:
    from PIL import Image, ImageDraw, ImageFont
    first = Image.open(rows[0][1][0])
    w, h = first.size
    n = max(len(r[1]) for r in rows)
    W = label_w + n * w + (n - 1) * gap
    H = len(rows) * h + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = None
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, size=max(18, h // 9))
            break
    for r, (label, tiles) in enumerate(rows):
        y = r * (h + gap)
        for k, p in enumerate(tiles):
            canvas.paste(Image.open(p).convert("RGB"), (label_w + k * (w + gap), y))
        if label_w > 0:
            tmp = Image.new("RGB", (h, label_w), (255, 255, 255))
            ImageDraw.Draw(tmp).text((h // 2, label_w // 2), label, fill=(20, 20, 20),
                                     font=font, anchor="mm")
            canvas.paste(tmp.rotate(90, expand=True), (0, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    canvas.save(out.with_suffix(".pdf"), resolution=300.0)
    logger.info("wrote %s (%dx%d)", out, W, H)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["phone_flip", "screwdriver_turn", "card_pickup"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--res", type=int, default=800)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--label-w", type=int, default=90)
    p.add_argument("--compose-only", action="store_true")
    p.add_argument("--view", nargs="*", default=[], metavar="TASK=DIST,AZIM,ELEV[,DZ]",
                   help="override a task's camera framing")
    p.add_argument("--only", type=int, nargs="*", default=None, help="render just these tile indices")
    p.add_argument("--tight", action="store_true", help="tight framing (VIEWS_TIGHT) with either style")
    p.add_argument("--style", choices=["paper", "rebuttal"], default="paper",
                   help="'rebuttal' draws paper tasks like the rebuttal figure (blue object, tight framing)")
    p.add_argument("--tag", default="", help="suffix for tile filenames (camera tests)")
    args = p.parse_args()
    _STYLE[0] = args.style
    if args.style == "rebuttal" or args.tight:
        VIEWS.update(VIEWS_TIGHT)
    for kv in args.view:
        t, vals = kv.split("=", 1)
        v = [float(x) for x in vals.split(",")]
        VIEWS[t] = dict(dist=v[0], azim=v[1], elev=v[2], lookat_off=(0.0, 0.0, v[3] if len(v) > 3 else 0.0))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tiles_dir = args.out.parent / "tiles"
    rows = []
    for task, label in TASKS:
        if task not in args.tasks:
            continue
        if args.compose_only:
            tiles = sorted(tiles_dir.glob(f"{task}_*.png"))
        else:
            tiles = render_task(task, args.replay_root / task, args.n, args.samples, args.res, tiles_dir,
                                args.only, args.tag)
        rows.append((label, tiles))
    compose(rows, args.out, args.gap, args.label_w)


if __name__ == "__main__":
    main()
