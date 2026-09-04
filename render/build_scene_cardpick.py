"""Build a Blender scene for a card_pick replay (xArm7 + LEAP hand + card stack).

Reuses `build_scene_pegclimb` for the body/geom pass, the LEAP/xArm materials and
the Cycles setup, and replaces two things that are wrong for this task:

* **Card materials.** The manipulated card is the base env's `peg` body, so the
  peg-climb dispatcher paints it chips-can light blue. Here the top card is a
  warm orange and the cards under it are cream, so "took ONE card off the stack"
  is legible at a glance.
* **Camera.** The model's `scene_cam` was framed for the palm-up cube task and
  leaves the hand in the far corner of a mostly empty table. This aims a camera
  at the stack from a three-quarter view, close enough to see the fingertips.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

import build_scene as bs
import build_scene_pegclimb as bsp

logger = logging.getLogger("build_scene_cardpick")

# Bind the original dispatcher NOW: main() swaps `bsp.material_for_geom` to this
# module's version, so a late lookup would resolve to itself and recurse.
_PEGCLIMB_MATERIAL_FOR_GEOM = bsp.material_for_geom

TOP_CARD_ORANGE = (0.86, 0.28, 0.14, 1.0)
STACK_CREAM = (0.80, 0.75, 0.66, 1.0)
PEDESTAL_GREY = (0.58, 0.60, 0.64, 1.0)


def _card_material(name, colour, rough):
    return bs.make_principled_material(
        name, base_color=colour, roughness=rough, metallic=0.0,
        specular=0.35)


def material_for_geom(body_name: str, geom: dict):
    bname = (body_name or "").lower()
    gname = (geom.get("name") or "").lower()

    # The TOP card first: its body is named `peg`, which the peg-climb
    # dispatcher would paint chips-can blue.
    if bname == "peg" or "cube_visual" in gname or "top_card" in gname:
        return _card_material("mat__card_top", TOP_CARD_ORANGE, 0.55)
    if bname.startswith("card_") and "table" not in bname:
        return _card_material("mat__card_stack", STACK_CREAM, 0.65)
    if "card_table" in bname or "card_post" in gname or "card_table" in gname:
        return _card_material("mat__pedestal", PEDESTAL_GREY, 0.5)
    return _PEGCLIMB_MATERIAL_FOR_GEOM(body_name, geom)


def add_lookat_camera(target, eye, res_x, res_y, fovy_deg):
    cam_data = bpy.data.cameras.new("cardpick_cam")
    cam = bpy.data.objects.new("cardpick_cam", cam_data)
    bpy.context.collection.objects.link(cam)
    eye_v, tgt_v = Vector(eye), Vector(target)
    cam.location = eye_v
    cam.rotation_mode = "QUATERNION"
    # Blender cameras look down -Z with +Y up.
    cam.rotation_quaternion = (eye_v - tgt_v).normalized().to_track_quat("Z", "Y")
    scn = bpy.context.scene
    scn.render.resolution_x = res_x
    scn.render.resolution_y = res_y
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(fovy_deg)
    scn.camera = cam
    logger.info("camera at %s looking at %s (fovy %.0f)", tuple(eye), tuple(target),
                fovy_deg)
    return cam


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--res-x", type=int, default=1280)
    p.add_argument("--res-y", type=int, default=720)
    p.add_argument("--floor-expand", type=float, default=4.0)
    p.add_argument("--eye", type=float, nargs=3, default=[1.02, 0.40, 0.56])
    p.add_argument("--target", type=float, nargs=3, default=[0.645, 0.00, 0.30])
    p.add_argument("--fovy", type=float, default=32.0)
    p.add_argument("--output-blend", required=True, type=Path)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()

    logging.basicConfig(level=getattr(logging, a.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    rd = a.replay_dir.resolve()
    scene = json.loads((rd / "scene.json").read_text())
    traj = dict(np.load(rd / "traj.npz", allow_pickle=True))

    bs.reset_scene()
    # Swap the material dispatcher the shared body pass calls.
    orig = bsp.material_for_geom
    bsp.material_for_geom = material_for_geom
    try:
        body_empties = bsp.build_bodies_and_geoms(scene, floor_expand=a.floor_expand)
    finally:
        bsp.material_for_geom = orig

    bs.keyframe_bodies(body_empties, traj, fps=a.fps, frame_step=a.frame_step,
                       time_scale=None)
    add_lookat_camera(a.target, a.eye, a.res_x, a.res_y, a.fovy)
    bs.setup_world_lighting(hdri_path=None, strength=0.6)
    bs.add_mjcf_style_lights()
    bs.configure_cycles(samples=a.samples, res_x=a.res_x, res_y=a.res_y)

    a.output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output_blend.resolve()))
    logger.info("saved blend: %s  (success=%s)", a.output_blend,
                bool(traj.get("success", False)))


if __name__ == "__main__":
    main()
