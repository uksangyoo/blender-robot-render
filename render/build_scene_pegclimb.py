"""Build a Blender scene for a PegClimb replay (xArm7 + LEAP hand + peg).

PegClimb analogue of ``build_scene.py``. It reuses that module's mesh import,
PBR-material, keyframe, lighting and Cycles helpers, but replaces the
LIBERO/Panda-specific body+material logic:

  * No mujoco_menagerie Panda substitution (our robot is an xArm7 whose body
    names — link1..link7 — would otherwise be mistaken for Franka links).
  * Materials are driven from the per-geom *effective* rgba exported by
    ``replay_peg_climb.py`` plus a few hand-tuned recipes:
      - peg (chips-can)  -> light-blue plastic (like the LIBERO crate boxes)
      - LEAP hand parts  -> matte black 3D-print plastic
      - xArm white links -> off-white PBR shell (subtle noise bump)
      - xArm end_tool    -> brushed grey metal
      - table floor      -> light studio surface

Runs under a Python 3.11 env with the ``bpy`` module (conda env
``blender-render``)::

    conda run -n blender-render python \\
        thirdparty/blender-robot-render/render/build_scene_pegclimb.py \\
        --replay-dir thirdparty/blender-robot-render/outputs/peg_climb_seed3 \\
        --output-blend thirdparty/blender-robot-render/outputs/peg_climb_seed3/scene.blend
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
from mathutils import Quaternion, Vector

# Make the sibling build_scene module importable (reuse its helpers).
_RENDER_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _RENDER_DIR.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
from render import build_scene as bs  # noqa: E402

logger = logging.getLogger("build_scene_pegclimb")


# ---------------------------------------------------------------------------
# Peg-climb materials
# ---------------------------------------------------------------------------

# Light blue, in the spirit of the LIBERO crate boxes but lighter.
PEG_LIGHT_BLUE = (0.30, 0.56, 0.82, 1.0)


def _peg_material():
    return bs.make_principled_material(
        "mat__peg_lightblue",
        base_color=PEG_LIGHT_BLUE,
        roughness=0.34,
        metallic=0.0,
        specular=0.5,
        noise_bump=0.12,
        noise_scale=240.0,
        roughness_variation=0.10,
    )


def _leap_black_material():
    # Matte black 3D-print plastic for the LEAP hand.
    return bs.make_principled_material(
        "mat__leap_black",
        base_color=(0.045, 0.045, 0.05, 1.0),
        roughness=0.55,
        metallic=0.0,
        specular=0.4,
        noise_bump=0.25,
        noise_scale=520.0,
        roughness_variation=0.12,
    )


def _xarm_white_material():
    # Off-white PBR shell — the subtle noise bump keeps the plastic from
    # reading as CGI-perfect (mirrors build_scene._panda_shell_material).
    return bs.make_principled_material(
        "mat__xarm_white",
        base_color=(0.93, 0.94, 0.96, 1.0),
        roughness=0.30,
        metallic=0.0,
        specular=0.55,
        noise_bump=0.16,
        noise_scale=420.0,
        roughness_variation=0.12,
    )


def _xarm_metal_material():
    # Brushed grey metal for the end-tool flange.
    return bs.make_principled_material(
        "mat__xarm_metal",
        base_color=(0.52, 0.53, 0.55, 1.0),
        roughness=0.38,
        metallic=0.55,
        specular=0.6,
        noise_bump=0.10,
        noise_scale=600.0,
        roughness_variation=0.15,
    )


def _floor_material():
    return bs.make_principled_material(
        "mat__pegclimb_floor",
        base_color=(0.82, 0.82, 0.85, 1.0),
        roughness=0.40,
        metallic=0.0,
        specular=0.4,
    )


def _fallback_material(name, rgba):
    return bs.make_principled_material(
        f"mat__{name}",
        base_color=tuple(rgba[:4]),
        roughness=0.45,
        metallic=0.0,
    )


def material_for_geom(body_name: str, geom: dict):
    """Pick a PBR material for one peg-climb geom."""
    bname = (body_name or "").lower()
    gname = (geom.get("name") or "").lower()
    mat_name = (geom.get("material") or "").lower()
    rgba = geom.get("rgba_geom") or [0.7, 0.7, 0.7, 1.0]

    if "peg" in bname or "peg" in gname or "chips" in (geom.get("mesh_name") or "").lower():
        return _peg_material()
    if bname.startswith("leap") or "leap" in mat_name:
        return _leap_black_material()
    if geom.get("type") == "plane" or "floor" in gname or "table" in gname:
        return _floor_material()
    if "end_tool" in (geom.get("mesh_name") or "").lower() or mat_name == "gray":
        return _xarm_metal_material()
    # xArm links default to the white shell.
    if bname.startswith("link") or mat_name == "white":
        return _xarm_white_material()
    return _fallback_material(f"{bname}_{gname or 'g'}", rgba)


# ---------------------------------------------------------------------------
# Build pass
# ---------------------------------------------------------------------------

def build_bodies_and_geoms(scene: dict, floor_expand: float) -> dict:
    """One Empty per body; visual geoms parented in with peg-climb materials."""
    body_empties = {}
    for body in scene["bodies"]:
        name = body["name"]
        empty = bpy.data.objects.new(name=f"body__{name}", object_data=None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.02
        bpy.context.collection.objects.link(empty)
        empty.rotation_mode = "QUATERNION"
        body_empties[name] = empty

    n_mesh = n_prim = n_fail = 0
    for body in scene["bodies"]:
        body_empty = body_empties[body["name"]]
        for geom in body["geoms"]:
            gtype = geom["type"]
            scale = (1.0, 1.0, 1.0)
            if gtype == "mesh":
                f = geom.get("mesh_file")
                if not f:
                    n_fail += 1
                    continue
                try:
                    mesh = bs.import_mesh(Path(f))
                    n_mesh += 1
                    scale = tuple(geom.get("mesh_scale") or (1.0, 1.0, 1.0))
                except Exception as e:
                    logger.warning("failed to import %s: %s", f, e)
                    n_fail += 1
                    continue
            else:
                mesh = bs._make_primitive_mesh(gtype, geom["size"])
                if mesh is None:
                    n_fail += 1
                    continue
                n_prim += 1
                scale = bs.primitive_scale(gtype, geom["size"])
                if gtype == "plane":
                    scale = (scale[0] * floor_expand, scale[1] * floor_expand, scale[2])

            obj = bpy.data.objects.new(
                name=f"geom__{body['name']}__{geom['name']}", object_data=mesh)
            bpy.context.collection.objects.link(obj)
            obj.parent = body_empty
            obj.location = Vector(geom["local_pos"])
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = bs.quat_wxyz(geom["local_quat"])
            obj.scale = scale

            # Per-object material (a mesh datablock may be shared across
            # several geoms — e.g. the 3 identical LEAP fingers — so assign
            # at object level rather than mutating the shared mesh).
            mat = material_for_geom(body["name"], geom)
            obj.data = obj.data.copy()  # unique mesh so material slots don't clash
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    logger.info("built %d bodies, %d meshes, %d primitives, %d failed",
                len(body_empties), n_mesh, n_prim, n_fail)
    return body_empties


def add_camera(cameras: dict, cam_name: str, res_x: int, res_y: int):
    """Create the scene camera from MuJoCo's static pose + vertical fovy."""
    entry = cameras.get(cam_name)
    if entry is None:
        raise ValueError(f"camera {cam_name!r} not in cameras.json")
    static = entry["static"]
    cam_data = bpy.data.cameras.new(name=cam_name)
    cam = bpy.data.objects.new(name=f"cam__{cam_name}", object_data=cam_data)
    bpy.context.collection.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    cam.location = Vector(static["local_pos"])
    cam.rotation_quaternion = bs.quat_wxyz(static["local_quat"])

    scn = bpy.context.scene
    scn.render.resolution_x = res_x
    scn.render.resolution_y = res_y
    # MuJoCo fovy is the *vertical* field of view; match it exactly.
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle_y = math.radians(float(static.get("fovy", 45.0)))
    scn.camera = cam
    logger.info("camera %s: fovy=%.1fdeg res=%dx%d pos=%s",
                cam_name, static.get("fovy"), res_x, res_y, static["local_pos"])
    return cam


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--camera", default="scene_cam")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--time-scale", type=float, default=None)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--res-x", type=int, default=1280)
    p.add_argument("--res-y", type=int, default=720)
    p.add_argument("--floor-expand", type=float, default=4.0,
                   help="scale up the table plane so it reads as a backdrop")
    p.add_argument("--hdri", type=Path, default=None)
    p.add_argument("--output-blend", required=True, type=Path)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    rd = args.replay_dir.resolve()
    scene = json.loads((rd / "scene.json").read_text())
    cameras = json.loads((rd / "cameras.json").read_text())
    traj = dict(np.load(rd / "traj.npz", allow_pickle=True))

    bs.reset_scene()
    body_empties = build_bodies_and_geoms(scene, floor_expand=args.floor_expand)
    bs.keyframe_bodies(body_empties, traj, fps=args.fps,
                       frame_step=args.frame_step, time_scale=args.time_scale)
    add_camera(cameras, args.camera, args.res_x, args.res_y)
    bs.setup_world_lighting(hdri_path=args.hdri, strength=0.6)
    bs.add_mjcf_style_lights()
    bs.configure_cycles(samples=args.samples, res_x=args.res_x, res_y=args.res_y)

    args.output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()))
    logger.info("saved blend: %s", args.output_blend)


if __name__ == "__main__":
    main()
