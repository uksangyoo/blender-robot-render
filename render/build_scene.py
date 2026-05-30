"""Build a Blender scene from a replay's scene.json + traj.npz.

Inputs: ``<out_dir>/scene.json`` and ``<out_dir>/traj.npz`` (produced by
``render/replay.py``).

The Blender scene is structured as a flat list of Empties — one per MuJoCo
body — with the visual geoms parented to them. Per-frame, we set each
empty's ``location`` and ``rotation_quaternion`` from ``traj.npz`` and
``keyframe_insert``. Cycles is configured with OPTIX (RTX) + OIDN
denoising; the camera matches the trial's logged ``agentview`` intrinsics.

Runs as a regular Python script under the local ``bpy`` env::

    cd ~/Projects/blender-robot-render
    uv run python render/build_scene.py \\
        --replay-dir outputs/trial_15 \\
        --camera agentview \\
        --output outputs/trial_15/scene.blend
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np

import bpy
from mathutils import Matrix, Quaternion, Vector

logger = logging.getLogger("build_scene")


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def reset_scene() -> None:
    """Wipe everything — meshes, materials, lights, cameras — for a clean build."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for collection in (
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.images,
        bpy.data.armatures,
    ):
        for item in list(collection):
            collection.remove(item)
    _mesh_cache.clear()
    _reset_material_cache()


def deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def quat_wxyz(arr) -> Quaternion:
    """Convert a [w, x, y, z] iterable to Blender's Quaternion (also wxyz)."""
    w, x, y, z = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
    return Quaternion((w, x, y, z))


# ---------------------------------------------------------------------------
# Mesh import
# ---------------------------------------------------------------------------

_mesh_cache: dict[str, bpy.types.Mesh] = {}


def _import_with_op(path: Path) -> list[bpy.types.Object]:
    """Run the appropriate Blender importer for ``path`` and return the new objects.

    Dispatches on extension because MJCF mesh assets are a mix of .obj
    (most of the Panda + crate scene) and .stl (gripper fingers / hand
    primitives). bpy 4.x ships separate operators; only .obj+.stl matter
    for our scene today.
    """
    ext = path.suffix.lower()
    existing = set(bpy.data.objects)
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif ext == ".stl":
        # Blender 4.x ships two STL operators — a deprecated Python one
        # (import_mesh.stl) and the new C++ one (wm.stl_import). Prefer
        # the new one when present.
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(path))
        else:
            bpy.ops.import_mesh.stl(filepath=str(path))
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh extension: {path}")
    return [o for o in bpy.data.objects if o not in existing]


def import_mesh(path: Path) -> bpy.types.Mesh:
    """Import any supported mesh file once; cache the mesh datablock by path.

    We deliberately throw away the placeholder Object(s) the importer creates
    and keep only the mesh datablock — every geom of the same MJCF mesh
    asset shares one ``bpy.data.meshes`` entry, which keeps the .blend
    small and lets us swap in per-geom materials cleanly.
    """
    key = str(path.resolve())
    if key in _mesh_cache:
        return _mesh_cache[key]

    if not path.exists():
        raise FileNotFoundError(f"mesh file not found: {path}")

    imported = _import_with_op(path)
    if not imported:
        raise RuntimeError(f"import created no objects: {path}")

    # Join multiple-object imports into one mesh so a single MJCF mesh
    # maps to a single Blender mesh datablock.
    if len(imported) > 1:
        for o in bpy.data.objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = imported[0]
        for obj in imported:
            obj.select_set(True)
        bpy.ops.object.join()
        merged = bpy.context.view_layer.objects.active
    else:
        merged = imported[0]

    mesh = merged.data
    bpy.data.objects.remove(merged, do_unlink=True)
    _mesh_cache[key] = mesh
    return mesh


def _make_primitive_mesh(geom_type: str, size: list[float]) -> bpy.types.Mesh | None:
    """Build a small primitive mesh for non-mesh MJCF geoms (box/sphere/cyl).

    These show up on the floor, walls, ceiling, etc. We make a tiny
    placeholder mesh that we'll instance with proper per-geom scaling.
    """
    if geom_type == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
    elif geom_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
    elif geom_type in {"cylinder", "capsule"}:
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=2.0, vertices=32)
    elif geom_type == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
    elif geom_type == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2.0)
    else:
        return None
    obj = bpy.context.active_object
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def primitive_scale(geom_type: str, size: list[float]) -> tuple[float, float, float]:
    """MJCF size semantics → Blender scale on the unit primitive above.

    https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-size
    """
    if geom_type == "box":          # size = half-extents
        return float(size[0]), float(size[1]), float(size[2])
    if geom_type == "sphere":       # size = radius
        r = float(size[0])
        return r, r, r
    if geom_type in {"cylinder", "capsule"}:  # size = (radius, half-length)
        r = float(size[0])
        h = float(size[1])
        return r, r, h
    if geom_type == "ellipsoid":
        return float(size[0]), float(size[1]), float(size[2])
    if geom_type == "plane":
        return float(size[0]), float(size[1]), 1.0
    return 1.0, 1.0, 1.0


# ---------------------------------------------------------------------------
# Material library
# ---------------------------------------------------------------------------

def make_principled_material(
    name: str,
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
    roughness: float = 0.45,
    metallic: float = 0.0,
    specular: float = 0.5,
    noise_bump: float = 0.0,
    noise_scale: float = 50.0,
    roughness_variation: float = 0.0,
) -> bpy.types.Material:
    """Build a Principled BSDF material with optional procedural detail.

    ``noise_bump`` adds a fine surface roughness via a Noise → Bump node
    chain. ``roughness_variation`` modulates the Roughness input with the
    same noise so the surface picks up subtle reflection variation under
    light, breaking up the otherwise plastic-perfect look that makes
    CG-rendered robots feel artificial. Both default to 0 (no effect).
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    spec_key = (
        "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    )
    bsdf.inputs[spec_key].default_value = specular
    bsdf.inputs["Roughness"].default_value = roughness

    if noise_bump > 0 or roughness_variation > 0:
        coord = nodes.new("ShaderNodeTexCoord")
        coord.location = (-600, 0)
        noise = nodes.new("ShaderNodeTexNoise")
        noise.location = (-300, 0)
        noise.inputs["Scale"].default_value = noise_scale
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.6
        links.new(coord.outputs["Object"], noise.inputs["Vector"])
        if noise_bump > 0:
            bump = nodes.new("ShaderNodeBump")
            bump.location = (0, -200)
            bump.inputs["Strength"].default_value = noise_bump
            bump.inputs["Distance"].default_value = 0.0005
            links.new(noise.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        if roughness_variation > 0:
            mix = nodes.new("ShaderNodeMix")
            mix.data_type = "FLOAT"
            mix.location = (0, 200)
            mix.inputs["Factor"].default_value = roughness_variation
            mix.inputs[2].default_value = roughness
            mix.inputs[3].default_value = min(1.0, roughness + 0.25)
            links.new(noise.outputs["Fac"], mix.inputs["Factor"])
            links.new(mix.outputs["Result"], bsdf.inputs["Roughness"])

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def _panda_finger_mat() -> bpy.types.Material:
    """Hard rubber black for the gripper finger pads."""
    return make_principled_material(
        "mat__panda_finger",
        base_color=(0.04, 0.04, 0.04, 1.0),
        roughness=0.62,
        metallic=0.0,
    )


def _panda_shell_material() -> bpy.types.Material:
    """Off-white PBR plastic for the main Panda shells.

    Subtle Noise bump + roughness modulation breaks up the otherwise
    plastic-perfect look so the surface picks up convincing micro-
    variation under the studio lights — the difference between this and
    a flat-colour shader is what makes the arm read as a real-world
    object rather than CGI.
    """
    return make_principled_material(
        "mat__panda_unified_shell",
        base_color=(0.93, 0.94, 0.96, 1.0),
        roughness=0.28,
        metallic=0.0,
        specular=0.55,
        noise_bump=0.18,
        noise_scale=420.0,
        roughness_variation=0.12,
    )


def _panda_accent_material() -> bpy.types.Material:
    """Mid-dark grey rubberised plastic for Panda joint collars.

    Previous version was too close to pure black, which made the joint
    accents read as patchy high-contrast smears on the white shells.
    The real Franka collars are a medium-dark plastic — closer to the
    MJCF's authored 0.25 luminance but with a soft, matte finish.
    Pairing a matte roughness with no metallic + minimal specular keeps
    the seam against the white shell from drawing the eye.
    """
    return make_principled_material(
        "mat__panda_accent",
        base_color=(0.20, 0.20, 0.22, 1.0),
        roughness=0.70,
        metallic=0.0,
        specular=0.35,
        noise_bump=0.30,
        noise_scale=600.0,
        roughness_variation=0.15,
    )


def _is_dark_accent_rgba(rgba) -> bool:
    """True when the MJCF authored an explicitly dark material colour.

    Filters out:
      - ``None`` (rgba not specified → use default white shell)
      - High-saturation debug colours (red/green) that aren't real visuals
      - Light colours (luminance ≥ 0.4)

    What's left is the small set of CAD parts the Panda designers
    actually wanted to render darker — the joint collars and motor caps.
    """
    if rgba is None or len(rgba) < 3:
        return False
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    sat = max(r, g, b) - min(r, g, b)
    if sat > 0.6:
        return False
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance < 0.40


def _looks_like_accent_geom(geom: dict) -> bool:
    """Heuristic: does this sub-mesh look like a Panda joint accent ring?

    A real Franka joint collar is a *thin disc / band* — short along one
    axis (the joint axial direction), wide in the other two. The MJCF
    ``geom_size`` array stores half-extents for primitives, but for mesh
    geoms it's all zeros, so we walk the imported mesh's bounding box.

    Returns True if the mesh's smallest axis is < 35 % of its largest
    axis (clearly disc-shaped) AND the geom isn't tiny (< 0.005 m on its
    longest axis — those are usually buttons / LEDs we don't want to
    drag dark across the shell). False otherwise.
    """
    file = geom.get("mesh_file")
    if not file:
        return False
    p = Path(file)
    if not p.exists():
        return False
    try:
        m = import_mesh(p)
    except Exception:
        return False
    if not m.vertices:
        return False
    xs = [v.co.x for v in m.vertices]
    ys = [v.co.y for v in m.vertices]
    zs = [v.co.z for v in m.vertices]
    dims = sorted(
        [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
        reverse=True,
    )
    if dims[0] < 0.005:
        return False
    ratio = dims[2] / dims[0]
    return ratio < 0.35


def _panda_material_from_rgba(
    mat_name: str,
    rgba: tuple[float, float, float, float] | None,
) -> bpy.types.Material:
    """Pick the right Panda PBR material from the MJCF rgba.

    Authored-dark → joint accent. Anything else (none, white, debug
    primaries) → the white shell. We rely on the caller to apply a
    small outward scale to dark-accent geoms so they don't Z-fight with
    the coincident white shell underneath.
    """
    if _is_dark_accent_rgba(rgba):
        return _panda_accent_material()
    return _panda_shell_material()


def derive_material_for_geom(geom: dict, body_name: str) -> bpy.types.Material:
    """Pick a Blender material for one MJCF geom.

    Heuristic order:
      1. Hand-tuned overrides keyed off the body name (panda link, gripper,
         crate machine, crate box, room).
      2. The MJCF material's rgba (when set) → Principled BSDF.
      3. The geom-level rgba override (when set).
      4. Default light grey.

    Material recipes:
      - Panda links / gripper hand: matte off-white plastic; dark "joint"
        accents would need per-sub-mesh material assignment (TODO).
      - Gripper fingers: hard black rubber.
      - Crate machine: light grey painted metal — low roughness for a
        slight shine, slightly desaturated to match the original.
      - Crate boxes: medium-saturation blue, plastic finish.
      - Floor: light grey with low roughness (gentle floor reflection).
      - Walls: matte dark grey to add visual contrast (the original eval
        render has dark walls that make the workspace pop).
    """
    bname = body_name.lower()
    gname = (geom.get("name") or "").lower()

    if "finger" in bname:
        return make_principled_material(
            "mat__gripper_finger",
            base_color=(0.04, 0.04, 0.04, 1.0),
            roughness=0.55,
            metallic=0.0,
        )
    if "gripper" in bname or "hand" in bname:
        # Panda gripper hand body — same off-white as the arm links.
        return make_principled_material(
            "mat__gripper_body",
            base_color=(0.88, 0.88, 0.88, 1.0),
            roughness=0.32,
            metallic=0.0,
            specular=0.55,
        )
    if "robot" in bname and ("link" in bname or "base" in bname or "mount" in bname):
        return make_principled_material(
            "mat__panda_link",
            base_color=(0.92, 0.92, 0.92, 1.0),
            roughness=0.30,
            metallic=0.0,
            specular=0.55,
        )
    if "crate_machine" in bname:
        # Dark industrial painted steel — base must be quite dark
        # because AgX tone mapping lifts mid-grey back toward white
        # under the bright sun lamps. ~0.10 linear reads as a
        # convincing "slate grey" in the final image without needing
        # to dim the rest of the scene.
        return make_principled_material(
            "mat__crate_machine",
            base_color=(0.10, 0.11, 0.13, 1.0),
            roughness=0.60,
            metallic=0.10,
            specular=0.4,
            noise_bump=0.8,
            noise_scale=320.0,
            roughness_variation=0.35,
        )
    if "crate_box" in bname:
        # Industrial blue plastic crate — subtle scuff via noise bump.
        return make_principled_material(
            "mat__crate_box",
            base_color=(0.14, 0.27, 0.52, 1.0),
            roughness=0.48,
            metallic=0.0,
            specular=0.5,
            noise_bump=0.3,
            noise_scale=180.0,
            roughness_variation=0.15,
        )
    if "robot_platform" in bname:
        # Dark cast concrete pedestal. Same brightness consideration as
        # ``crate_machine`` — the value has to be quite low for AgX to
        # render it as actual mid-dark grey rather than light grey.
        return make_principled_material(
            "mat__robot_platform",
            base_color=(0.08, 0.08, 0.09, 1.0),
            roughness=0.90,
            metallic=0.0,
            noise_bump=0.7,
            noise_scale=180.0,
            roughness_variation=0.30,
        )
    if bname == "world":
        mat = geom.get("material") or ""
        if "floor" in mat:
            return make_principled_material(
                "mat__floor",
                base_color=(0.78, 0.78, 0.80, 1.0),
                roughness=0.42,
                metallic=0.0,
            )
        if "ceiling" in mat:
            return make_principled_material(
                "mat__ceiling",
                base_color=(0.93, 0.93, 0.93, 1.0),
                roughness=0.85,
                metallic=0.0,
            )
        # All non-floor/ceiling world geoms — walls + signs — get the
        # darker "studio backdrop" treatment so the workspace pops.
        return make_principled_material(
            "mat__wall",
            base_color=(0.32, 0.34, 0.38, 1.0),
            roughness=0.78,
            metallic=0.0,
        )

    rgba = geom.get("rgba_geom")
    if rgba is None:
        m_props = geom.get("material_props") or {}
        rgba = m_props.get("rgba")
    if rgba is None or len(rgba) < 4:
        rgba = (0.7, 0.7, 0.7, 1.0)
    return make_principled_material(
        f"mat__{body_name}_{gname or 'g'}",
        base_color=tuple(rgba[:4]),
        roughness=0.45,
        metallic=0.0,
    )


# ---------------------------------------------------------------------------
# Build pass: bodies + geoms
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mujoco Menagerie Panda — clean visual assets
# ---------------------------------------------------------------------------
# robosuite's Panda meshes split each link into many CAD parts with mixed /
# inconsistent material colours, and the parts are coincident in 3D, which
# made every materials strategy we tried look patchy. mujoco_menagerie ships
# a hand-curated Panda where each sub-mesh has an explicit material
# (white / off_white / black / light_blue / green), with no coincident
# duplicates. Substituting menagerie's visual meshes into our scene gives
# the Franka its proper black-on-white look.
#
# Menagerie's per-body geom list — derived directly from
# ``mujoco_menagerie/franka_emika_panda/{panda,hand}.xml``. Order matters
# (later geoms paint over earlier ones during the rasteriser pass).

MENAGERIE_PANDA_ASSETS = Path(
    "/home/uyoo/Projects/MoVLA/third-party/mujoco_menagerie/franka_emika_panda/assets"
)

_MENAGERIE_PANDA_GEOMS: dict[str, list[tuple[str, str]]] = {
    "link0": [
        ("link0_0", "off_white"), ("link0_1", "black"), ("link0_2", "off_white"),
        ("link0_3", "black"), ("link0_4", "off_white"), ("link0_5", "black"),
        ("link0_7", "white"), ("link0_8", "white"), ("link0_9", "black"),
        ("link0_10", "off_white"), ("link0_11", "white"),
    ],
    "link1": [("link1", "white")],
    "link2": [("link2", "white")],
    "link3": [
        ("link3_0", "white"), ("link3_1", "white"),
        ("link3_2", "white"), ("link3_3", "black"),
    ],
    "link4": [
        ("link4_0", "white"), ("link4_1", "white"),
        ("link4_2", "black"), ("link4_3", "white"),
    ],
    "link5": [("link5_0", "black"), ("link5_1", "white"), ("link5_2", "white")],
    "link6": [
        ("link6_0", "off_white"), ("link6_1", "white"), ("link6_2", "black"),
        ("link6_3", "white"), ("link6_4", "white"), ("link6_5", "white"),
        ("link6_6", "white"), ("link6_7", "light_blue"), ("link6_8", "light_blue"),
        ("link6_9", "black"), ("link6_10", "black"), ("link6_11", "white"),
        ("link6_12", "green"), ("link6_13", "white"), ("link6_14", "black"),
        ("link6_15", "black"), ("link6_16", "white"),
    ],
    "link7": [
        ("link7_0", "white"), ("link7_1", "black"), ("link7_2", "black"),
        ("link7_3", "black"), ("link7_4", "black"), ("link7_5", "black"),
        ("link7_6", "black"), ("link7_7", "white"),
    ],
    "hand": [
        ("hand_0", "off_white"), ("hand_1", "black"), ("hand_2", "black"),
        ("hand_3", "white"), ("hand_4", "off_white"),
    ],
    "left_finger":  [("finger_0", "off_white"), ("finger_1", "black")],
    "right_finger": [("finger_0", "off_white"), ("finger_1", "black")],
}


def _menagerie_panda_link(body_name: str) -> str | None:
    """Map a robosuite Panda body name to menagerie's link/hand name.

    Returns ``None`` for non-Panda bodies, which keeps the existing
    robosuite-mesh-based path intact for everything else.

    Important: ``robot{N}_right_hand`` is a *kinematic* frame between
    link7 and the gripper body — it carries no visual mesh in robosuite,
    and applying menagerie's hand mesh here would double the gripper at
    a 90° offset (its world transform differs from the actual gripper's
    by the gripper-mount quat). The visual gripper lives on
    ``gripper{N}_right_gripper``; only match that.
    """
    b = body_name
    for suffix in ("link0", "link1", "link2", "link3",
                   "link4", "link5", "link6", "link7"):
        if b.endswith(suffix):
            return suffix
    if b.endswith("_right_gripper"):
        return "hand"
    if b.endswith("leftfinger"):
        return "left_finger"
    if b.endswith("rightfinger"):
        return "right_finger"
    return None


def _menagerie_body_correction(
    body_name: str,
    body_geoms: list[dict],
) -> tuple[float, float, float, float]:
    """Look up the local quat to apply to menagerie meshes on this body.

    Robosuite's gripper hand + right-finger geoms carry a non-identity
    user quat in their MJCF (90°Z and 180°Z respectively) — the visual
    only looks right after that rotation. Menagerie's hand/finger meshes
    were authored expecting the *body* to carry that rotation, not the
    geom, so when we attach them to robosuite bodies (which sit at
    identity) we have to recreate the rotation here.

    We read the existing robosuite geom's ``local_quat`` from the
    scene graph rather than hard-coding values — that way any future
    robot variant whose conventions differ still works automatically.
    Falls back to identity for bodies (e.g. the arm links) where the
    robosuite geom is already at identity.
    """
    for g in body_geoms:
        if g.get("type") != "mesh":
            continue
        if g.get("group", 2) > 2:
            continue
        gname = (g.get("name") or "").lower()
        if "visual" not in gname:
            continue
        q = g.get("local_quat") or [1.0, 0.0, 0.0, 0.0]
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return (1.0, 0.0, 0.0, 0.0)


def _menagerie_material(name: str) -> bpy.types.Material:
    """Build a PBR material matching one of menagerie's named materials."""
    palette = {
        "white":      ((0.95, 0.96, 0.97, 1.0), 0.30, 0.0),
        "off_white":  ((0.88, 0.90, 0.92, 1.0), 0.32, 0.0),
        "black":      ((0.06, 0.06, 0.07, 1.0), 0.55, 0.0),
        "green":      ((0.05, 0.55, 0.10, 1.0), 0.45, 0.0),
        "light_blue": ((0.04, 0.54, 0.78, 1.0), 0.45, 0.0),
    }
    base_color, roughness, metallic = palette.get(name, palette["white"])
    # Subtle noise bump on white / off_white so the plastic doesn't read as
    # CGI-perfect. Black accents stay clean.
    bump = 0.18 if name in ("white", "off_white") else 0.0
    return make_principled_material(
        f"mat__menagerie_{name}",
        base_color=base_color,
        roughness=roughness,
        metallic=metallic,
        specular=0.55,
        noise_bump=bump,
        noise_scale=420.0,
        roughness_variation=0.10 if bump > 0 else 0.0,
    )


_materials_assigned_meshes: set[int] = set()


def _is_panda_body(body_name: str) -> bool:
    bname = body_name.lower()
    return (
        ("robot" in bname and ("link" in bname or "base" in bname))
        or "hand" in bname
        or ("gripper" in bname and "finger" not in bname)
    )


def _assign_materials_to_mesh(
    mesh: bpy.types.Mesh,
    body_name: str,
    geom: dict,
) -> None:
    """Replace each material slot on ``mesh`` with a PBR material.

    For non-Panda bodies only — Panda bodies are handled by the
    menagerie-substitution path in ``build_bodies_and_geoms`` and never
    reach this function. We keep this single-material-per-mesh fallback
    for crates, the machine, the platform, and the room shell.
    """
    mesh_id = id(mesh)
    if mesh_id in _materials_assigned_meshes:
        return
    _materials_assigned_meshes.add(mesh_id)

    mat = derive_material_for_geom(geom, body_name)
    mesh.materials.clear()
    mesh.materials.append(mat)


def _reset_material_cache() -> None:
    """Called from reset_scene so mesh re-imports re-assign materials."""
    _materials_assigned_meshes.clear()


_NO_SHADOW_GEOMS = {
    "ceiling",        # always blocks the sun lamps in Cycles
    "wall_xpos", "wall_xneg", "wall_ypos", "wall_yneg",
    # The "bosch_sign" geoms are co-planar with the walls; they also
    # block illumination and we don't render the room shell anyway.
    "bosch_sign_xpos", "bosch_sign_xneg",
    "bosch_sign_ypos", "bosch_sign_yneg",
}


def build_bodies_and_geoms(scene: dict) -> dict[str, bpy.types.Object]:
    """Create one Empty per body and one Mesh-instance Object per visual geom.

    Returns a ``{body_name: bpy.Object}`` map keyed on the MJCF body name —
    the animate pass will key the per-frame transforms off this dict.

    Walls + ceiling are kept in the scene (so the floor and back-wall
    bounce light appropriately) but their ``visible_shadow`` Cycles flag
    is cleared so Blender sun lamps can illuminate the interior without
    being blocked from above/outside. Without this, Cycles renders pitch
    black while EEVEE looks fine.
    """
    body_empties: dict[str, bpy.types.Object] = {}

    # Pass 1: create all empties
    for body in scene["bodies"]:
        name = body["name"]
        empty = bpy.data.objects.new(name=f"body__{name}", object_data=None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.02
        bpy.context.collection.objects.link(empty)
        empty.rotation_mode = "QUATERNION"
        body_empties[name] = empty

    # Pass 2: load meshes + parent geoms to empties
    n_meshes = 0
    n_prims = 0
    n_failed = 0
    n_menagerie = 0
    for body in scene["bodies"]:
        body_empty = body_empties[body["name"]]

        # Hijack Panda bodies and inject mujoco_menagerie's clean visual
        # meshes instead of robosuite's CAD-detail-heavy ones. Body
        # kinematics still come from the replay; only the per-link
        # visual is swapped.
        #
        # The frame conventions differ between the two model sources for
        # the gripper hand and its fingers: robosuite applies a 90°Z to
        # the hand mesh and a 180°Z mirror to the right finger mesh via
        # the geom's local quat (its hand/finger bodies sit at identity
        # orientation). Menagerie expects the mesh at identity in a
        # body that's *already* in the mirrored orientation. To preserve
        # the user-visible visual we need to reproduce robosuite's
        # local-quat by hand when placing menagerie meshes — otherwise
        # the gripper appears 90° off and the right finger looks
        # unmirrored.
        menagerie_link = _menagerie_panda_link(body["name"])
        if menagerie_link is not None and menagerie_link in _MENAGERIE_PANDA_GEOMS:
            local_quat = _menagerie_body_correction(body["name"], body.get("geoms", []))
            for mesh_basename, mat_name in _MENAGERIE_PANDA_GEOMS[menagerie_link]:
                file = MENAGERIE_PANDA_ASSETS / f"{mesh_basename}.obj"
                if not file.exists():
                    logger.warning("menagerie mesh missing: %s", file)
                    continue
                try:
                    mesh = import_mesh(file)
                except Exception as e:
                    logger.warning("failed to import menagerie %s: %s", file, e)
                    continue
                obj = bpy.data.objects.new(
                    name=f"geom__{body['name']}__menagerie_{mesh_basename}",
                    object_data=mesh,
                )
                bpy.context.collection.objects.link(obj)
                obj.parent = body_empty
                obj.location = Vector((0, 0, 0))
                obj.rotation_mode = "QUATERNION"
                obj.rotation_quaternion = Quaternion(local_quat)
                obj.scale = (1.0, 1.0, 1.0)
                # Force per-object material (don't share via mesh) so the
                # same finger_0.obj on left vs right finger could carry
                # different mats if we ever needed it.
                obj.data.materials.clear()
                obj.data.materials.append(_menagerie_material(mat_name))
                n_menagerie += 1
            continue  # Skip the normal robosuite-mesh path for this body

        for geom in body["geoms"]:
            gtype = geom["type"]
            mesh = None
            scale = (1.0, 1.0, 1.0)
            if gtype == "mesh":
                file = geom.get("mesh_file")
                if not file:
                    logger.warning("geom %s has type=mesh but no mesh_file", geom["name"])
                    n_failed += 1
                    continue
                try:
                    mesh = import_mesh(Path(file))
                    n_meshes += 1
                    scale = tuple(geom.get("mesh_scale") or (1.0, 1.0, 1.0))
                except Exception as e:
                    logger.warning("failed to import %s: %s", file, e)
                    n_failed += 1
                    continue
            else:
                mesh = _make_primitive_mesh(gtype, geom["size"])
                if mesh is None:
                    n_failed += 1
                    continue
                n_prims += 1
                scale = primitive_scale(gtype, geom["size"])

            obj = bpy.data.objects.new(
                name=f"geom__{body['name']}__{geom['name']}",
                object_data=mesh,
            )
            bpy.context.collection.objects.link(obj)
            obj.parent = body_empty
            obj.location = Vector(geom["local_pos"])
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = quat_wxyz(geom["local_quat"])

            obj.scale = scale
            # Panda bodies were handled by the menagerie branch above.
            # For everything else use the body-name-based material.
            _assign_materials_to_mesh(mesh, body["name"], geom)

            # Skip shadow casting on the room shell so sun lamps reach inside.
            if geom.get("name") in _NO_SHADOW_GEOMS:
                obj.visible_shadow = False

    logger.info(
        "built %d bodies, %d meshes, %d primitives, %d failed",
        len(body_empties), n_meshes, n_prims, n_failed,
    )
    return body_empties


# ---------------------------------------------------------------------------
# Animation: keyframe each body
# ---------------------------------------------------------------------------

def keyframe_bodies(
    body_empties: dict[str, bpy.types.Object],
    traj: dict,
    fps: int = 24,
    frame_step: int = 1,
    time_scale: float | None = None,
) -> int:
    """Insert location + rotation_quaternion keyframes per body per frame.

    ``traj`` is the loaded ``traj.npz`` dict-like. By default, every replay
    sample is placed on the next Blender frame. When ``time_scale`` is set,
    logged timestamps are used instead: an 80-second replay at
    ``time_scale=8`` becomes a 10-second Blender animation.
    """
    body_pos = traj["body_pos"]       # (T, NB, 3)
    body_quat = traj["body_quat"]     # (T, NB, 4) wxyz
    body_names = list(traj["body_names"])
    T = body_pos.shape[0]

    scn = bpy.context.scene
    scn.render.fps = fps
    scn.frame_start = 1

    timestamps = traj.get("timestamps")
    if time_scale is not None:
        if timestamps is None:
            raise ValueError("--time-scale requires timestamps in traj.npz")
        if time_scale <= 0:
            raise ValueError("--time-scale must be positive")
        timestamps = np.asarray(timestamps, dtype=np.float64)
        t0 = float(timestamps[0])
    else:
        timestamps = None

    end_frame = 1
    for t in range(0, T, frame_step):
        if timestamps is None:
            frame = (t // frame_step) + 1
        else:
            frame = int(round(((float(timestamps[t]) - t0) / time_scale) * fps)) + 1
        for bi, bname in enumerate(body_names):
            obj = body_empties.get(str(bname))
            if obj is None:
                continue
            obj.location = Vector([float(x) for x in body_pos[t, bi]])
            obj.rotation_quaternion = quat_wxyz(body_quat[t, bi])
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        end_frame = frame

    for obj in body_empties.values():
        if obj.animation_data is None or obj.animation_data.action is None:
            continue
        for fcurve in obj.animation_data.action.fcurves:
            for key in fcurve.keyframe_points:
                key.interpolation = "LINEAR"

    scn.frame_end = end_frame
    logger.info(
        "inserted keyframes for %d bodies over %d frames%s",
        len(body_empties),
        end_frame,
        f" (time_scale={time_scale:g}x)" if time_scale is not None else "",
    )
    return end_frame


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def add_camera_from_log(
    cameras_json: dict,
    cam_name: str,
    body_empties: dict[str, bpy.types.Object] | None = None,
) -> bpy.types.Object:
    """Set up a Blender camera using the camera's MJCF static pose + logged intrinsics.

    The static MJCF pose is the camera *in its parent body's local frame*.
    For cameras attached to ``world`` (e.g. ``agentview``) that's already
    the world pose. For body-attached cameras (e.g. ``crate_topdown_view``
    on ``crate_topdown_view_mount``) we parent the Blender camera to the
    body's animated empty so it inherits the body's world transform.

    Orientation follows MuJoCo's camera convention (forward = -Z, up = +Y),
    which is identical to Blender's, so we plug the quaternion through
    without conversion.

    Intrinsics come from the per-frame log (``intrinsics`` 3x3) — we
    derive the focal length in mm from ``K[0,0]`` and the image width
    so the framing matches what the eval saw. The logged ``poses`` are
    *not* used directly: they're stored in robot0-base frame with the
    OpenCV convention (forward = +Z) which would require a multi-step
    conversion to map back to Blender.
    """
    entry = cameras_json.get(cam_name)
    if entry is None:
        raise ValueError(f"camera '{cam_name}' not in cameras.json")

    cam_data = bpy.data.cameras.new(name=cam_name)
    cam = bpy.data.objects.new(name=f"cam__{cam_name}", object_data=cam_data)
    bpy.context.collection.objects.link(cam)
    cam.rotation_mode = "QUATERNION"

    static = entry.get("static") or {}
    loc = static.get("local_pos", [0, 0, 1])
    quat = static.get("local_quat", [1, 0, 0, 0])
    cam.location = Vector(loc)
    cam.rotation_quaternion = quat_wxyz(quat)

    # Parent the camera to its MJCF body so animated bodies (e.g. wrist
    # cameras) carry it along. For body=world this is effectively a no-op
    # since the world empty is at origin with identity rotation.
    body_name = static.get("body_name") or "world"
    if body_empties is not None and body_name in body_empties:
        cam.parent = body_empties[body_name]

    # Intrinsics -----------------------------------------------------------
    scn = bpy.context.scene
    intr = None
    if "intrinsics" in entry:
        intr = np.asarray(entry["intrinsics"], dtype=np.float64)

    if intr is not None and intr.shape == (3, 3):
        fx = float(intr[0, 0])
        cx = float(intr[0, 2])
        cy = float(intr[1, 2])
        # Principal point is centred for all MuJoCo cameras, so 2*cx ≈ W.
        W = int(round(cx * 2)) or scn.render.resolution_x
        H = int(round(cy * 2)) or scn.render.resolution_y
        scn.render.resolution_x = W
        scn.render.resolution_y = H
        sensor_w_mm = 36.0  # Blender default 35 mm sensor full-frame
        cam_data.sensor_width = sensor_w_mm
        cam_data.lens = fx * sensor_w_mm / W
        cam_data.sensor_fit = "HORIZONTAL"
        logger.info(
            "camera %s: W=%d H=%d focal=%.2fmm (fx=%.1f), pos=%s",
            cam_name, W, H, cam_data.lens, fx, list(loc),
        )
    else:
        fovy_deg = float(static.get("fovy", 45.0))
        cam_data.angle = math.radians(fovy_deg)
        logger.info("camera %s: fovy=%.1fdeg pos=%s", cam_name, fovy_deg, list(loc))

    scn.camera = cam
    return cam


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

def setup_world_lighting(strength: float = 0.3, hdri_path: Path | None = None) -> None:
    """Set up the world background — HDRI if provided, else a soft tonemapped grey.

    Hand-tuned to look like a soft studio environment: a low-strength HDRI
    contributes most of the ambient bounce, supplemented by area lights in
    ``add_three_point_lights``.
    """
    scn = bpy.context.scene
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (300, 0)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)

    if hdri_path and hdri_path.exists():
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.location = (-300, 0)
        env.image = bpy.data.images.load(str(hdri_path))
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        logger.info("HDRI: %s", hdri_path)
    else:
        bg.inputs["Color"].default_value = (0.9, 0.92, 0.95, 1.0)
        logger.info("no HDRI; using flat sky color")

    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_mjcf_style_lights() -> None:
    """Recreate the MJCF's <light> rig: three directional lights from above.

    The original eval render used MuJoCo's MJCF lights:

        <light pos="0 0 2.5"     dir="0 0 -1"        directional="true" diffuse=".70 .70 .70"/>
        <light pos="1.0 -1.0 2.5" dir="-0.4 0.4 -1"  directional="true" diffuse=".50 .50 .50"/>
        <light pos="2.5 0 1.2"    dir="-1 0 -0.1"    directional="true" diffuse=".25 .25 .25"/>

    Plus a bright headlight (ambient .88, diffuse .45 from the camera).
    In Blender we use sun lamps for the directional + an HDRI / world
    background for the ambient — closer to the MJCF aesthetic and far
    less likely to blow out white materials than the previous three-point
    area rig.
    """
    def add_sun(name, pos, direction, strength):
        d = bpy.data.lights.new(name=name, type="SUN")
        d.energy = strength
        d.angle = 0.05  # softer shadow penumbra
        d.color = (1.0, 0.98, 0.96)
        o = bpy.data.objects.new(name=name, object_data=d)
        bpy.context.collection.objects.link(o)
        o.location = Vector(pos)
        # Sun lamp default aims -Z; rotate -Z to match direction
        dir_v = Vector(direction).normalized()
        z = Vector((0, 0, -1))
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = z.rotation_difference(dir_v)
        return o

    add_sun("KeyTop",       (0.0,  0.0, 2.5), (0.0,  0.0, -1.0), strength=2.8)
    add_sun("KeyTopSide",   (1.0, -1.0, 2.5), (-0.4, 0.4, -1.0), strength=2.0)
    add_sun("FillSide",     (2.5,  0.0, 1.2), (-1.0, 0.0, -0.1), strength=1.0)


# ---------------------------------------------------------------------------
# Cycles render config
# ---------------------------------------------------------------------------

def configure_cycles(
    samples: int = 256,
    denoiser: str = "OPTIX",
    res_x: int | None = None,
    res_y: int | None = None,
) -> None:
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "GPU"
    scn.cycles.samples = samples
    scn.cycles.adaptive_threshold = 0.01
    scn.cycles.use_denoising = True
    scn.cycles.denoiser = denoiser  # "OPTIX" or "OPENIMAGEDENOISE"
    scn.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.tile_size = 2048
    if res_x is not None:
        scn.render.resolution_x = res_x
    if res_y is not None:
        scn.render.resolution_y = res_y
    # AgX gives a softer roll-off than Filmic, less likely to clip the
    # near-white plastic. Exposure 0 = neutral; positive = brighter.
    scn.view_settings.view_transform = "AgX"
    scn.view_settings.look = "AgX - Base Contrast"
    scn.view_settings.exposure = 0.0

    # Tell Cycles to use the OPTIX devices we probed earlier.
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = (d.type != "CPU")

    logger.info(
        "Cycles: %d samples, denoiser=%s, res=%dx%d",
        samples, denoiser, scn.render.resolution_x, scn.render.resolution_y,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--camera", default="agentview")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help=(
            "Use traj timestamps to time-compress the animation by this factor "
            "(e.g. 8 makes an 80s replay render as 10s at the requested fps)."
        ),
    )
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--res-x", type=int, default=None)
    p.add_argument("--res-y", type=int, default=None)
    p.add_argument("--hdri", type=Path, default=None)
    p.add_argument("--output-blend", type=Path, default=None,
                   help="If set, save the assembled .blend here")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rd = args.replay_dir.resolve()
    scene = json.loads((rd / "scene.json").read_text())
    cameras = json.loads((rd / "cameras.json").read_text())
    traj = dict(np.load(rd / "traj.npz", allow_pickle=True))

    reset_scene()
    body_empties = build_bodies_and_geoms(scene)
    keyframe_bodies(
        body_empties,
        traj,
        fps=args.fps,
        frame_step=args.frame_step,
        time_scale=args.time_scale,
    )
    add_camera_from_log(cameras, args.camera, body_empties=body_empties)
    # World background — when no HDRI is supplied we mirror MuJoCo's
    # headlight: a flat near-white ambient with low strength so the
    # directional sun lamps dominate the lighting balance.
    setup_world_lighting(hdri_path=args.hdri, strength=0.6)
    add_mjcf_style_lights()
    configure_cycles(samples=args.samples, res_x=args.res_x, res_y=args.res_y)

    if args.output_blend is not None:
        args.output_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()))
        logger.info("saved blend: %s", args.output_blend)


if __name__ == "__main__":
    main()
