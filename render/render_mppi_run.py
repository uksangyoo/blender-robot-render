"""Render a real-rig MPPI hose-routing run in Blender, side by side with the rig's own film.

Stage 3 of ``hose-routing/scripts/v5/render_mppi_run.py`` (which runs all three stages). It consumes the
bundle the first two stages wrote -- ``robot_scene.json`` / ``robot_tracks.npz`` (the YAM arms, from the
rig's IK) and ``timeline.json`` / ``timeline.npz`` (every output frame: hose nodes, arm body poses, peg-side
status, camera, the film frame and run clock for the RGB panel, and the animated MPPI drawables) -- and
does no planning-side reasoning of its own: it is a renderer of that timeline.

    .venv/bin/python render/render_mppi_run.py \\
        --bundle ~/Projects/hose-routing/outputs/mppi_real_v5/run_20260917_150620/blender \\
        --out-dir ~/Projects/hose-routing/outputs/mppi_real_v5/run_20260917_150620/blender/render \\
        --output ~/Projects/hose-routing/outputs/mppi_real_v5/run_20260917_150620/blender/mppi_run.mp4

Useful while iterating: ``--frames 900:1200`` renders one planning cycle, ``--preview`` renders at half
size with 16 samples, ``--resume`` keeps frames already on disk, ``--compose-only`` re-encodes the
side-by-side from rendered frames (caption changes cost seconds, not a re-render), ``--save-blend`` writes
the assembled scene for inspection in the Blender UI.

``--blender-scene base.blend`` starts from an existing scene instead of the built-in dark studio: its
world, lights and any object named ``floor`` are kept, and only the run's content (arms, hose, pegs,
goal markers, MPPI drawables, camera) is added.

The look follows ``chair upright.mp4``: black studio, glossy floor with a sparse "+" grid, the manipulated
object solid and light, exploration samples as many thin translucent strokes, the chosen sample in one
reserved colour, a short caption in the lower-left of the render.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import bpy
from mathutils import Vector

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
from render import build_scene as bs  # noqa: E402
from render import build_scene_pegclimb as bsp  # noqa: E402

logger = logging.getLogger("render_mppi_run")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OK_RGB = (0.20, 0.83, 0.60)
BAD_RGB = (0.97, 0.36, 0.36)
PEG_RGB = (0.89, 0.63, 0.17)
HOSE_RGB = (0.86, 0.86, 0.84)
SEGMENT_RGB = dict(intro=(70, 70, 76), probe=(72, 160, 210), zfit=(70, 70, 76), plan=(33, 145, 140),
                   exec=(255, 51, 158), result=(52, 211, 153), outro=(70, 70, 76))


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def catmull_rom(points: np.ndarray, per_segment: int = 4) -> np.ndarray:
    p = np.asarray(points, np.float64)
    if len(p) < 3:
        return p
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    s = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        out.append(.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s ** 2
                         + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3))
    out.append(p[-1:])
    return np.vstack(out)


def hose_polyline(centres: np.ndarray) -> np.ndarray:
    """28 node centres -> the tube's centreline, ends extended by half a segment (showcase_scene.polyline)."""
    c = np.asarray(centres, np.float64)
    mid = .5 * (c[:-1] + c[1:])
    return np.vstack([c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])])


def linear(rgb):
    """sRGB display values -> the linear values Blender shaders expect (obj.color is read as linear light;
    passing sRGB straight in drew every colour lighter and greyer: green came out mint, red salmon)."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)


def interp_keys(keys, t: float, default: float = 1.0) -> float:
    if not keys:
        return default
    ts = [k[0] for k in keys]
    vs = [k[1] for k in keys]
    return float(np.interp(t, ts, vs))


def new_curve(name: str, points: np.ndarray, radius: float, material, resolution: int = 3):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = resolution
    cu.use_fill_caps = True
    cu.bevel_factor_mapping_end = "SPLINE"
    sp = cu.splines.new("POLY")
    sp.points.add(len(points) - 1)
    sp.use_smooth = True
    for p, v in zip(sp.points, points):
        p.co = (float(v[0]), float(v[1]), float(v[2]), 1.0)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    cu.materials.append(material)
    return obj


def set_curve_points(obj, points: np.ndarray) -> None:
    for p, v in zip(obj.data.splines[0].points, points):
        p.co = (float(v[0]), float(v[1]), float(v[2]), 1.0)


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def object_colour_material(name: str, roughness: float = 0.35, emission_prop: str = "emit"):
    """Base colour + alpha from the OBJECT's colour, emission strength from its custom property.

    One material serves hundreds of candidate strokes: each object sets ``obj.color`` (RGBA) and
    ``obj["emit"]`` per frame, which is what lets a frame loop recolour and fade them cheaply.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    info = nt.nodes.new("ShaderNodeObjectInfo")
    emit = nt.nodes.new("ShaderNodeAttribute")
    emit.attribute_type = "OBJECT"
    emit.attribute_name = emission_prop
    bsdf.inputs["Roughness"].default_value = roughness
    nt.links.new(info.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(info.outputs["Alpha"], bsdf.inputs["Alpha"])
    nt.links.new(info.outputs["Color"], bsdf.inputs["Emission Color"])
    # Principled's Alpha does not fade its emission (measured: an emissive stroke at alpha 0.08 still
    # glowed at ~60% of full), so the strength is scaled by the same alpha.
    fade = nt.nodes.new("ShaderNodeMath")
    fade.operation = "MULTIPLY"
    nt.links.new(emit.outputs["Fac"], fade.inputs[0])
    nt.links.new(info.outputs["Alpha"], fade.inputs[1])
    nt.links.new(fade.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.blend_method = "BLEND"
    return mat


def hose_material(length_m: float):
    """Off-white corrugated hose: a ring bump every ~11 mm along the tube (a curve's UV runs along it)."""
    mat = bs.make_principled_material("mat__hose", base_color=HOSE_RGB + (1.0,), roughness=0.42, specular=0.45)
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 2 * math.pi * max(20.0, length_m / 0.011)
    sin = nt.nodes.new("ShaderNodeMath")
    sin.operation = "SINE"
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.55
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(coord.outputs["UV"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["X"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], sin.inputs[0])
    nt.links.new(sin.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def floor_material(spacing=0.10, half_len=0.0075, half_width=0.0008):
    """Glossy black with a sparse grid of small "+" marks, like the reference render."""
    mat = bpy.data.materials.new("mat__floor_grid")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    L = nt.links
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.13
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(coord.outputs["Object"], sep.inputs["Vector"])

    def math_node(op, a, b=None):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                L.new(v, n.inputs[i])
        return n.outputs["Value"]

    def dist_to_grid(v):
        x = math_node("DIVIDE", v, spacing)
        x = math_node("ADD", x, 0.5)
        x = math_node("FRACT", x)
        x = math_node("SUBTRACT", x, 0.5)
        x = math_node("ABSOLUTE", x)
        return math_node("MULTIPLY", x, spacing)

    dx, dy = dist_to_grid(sep.outputs["X"]), dist_to_grid(sep.outputs["Y"])
    arm_a = math_node("MULTIPLY", math_node("LESS_THAN", dx, half_width), math_node("LESS_THAN", dy, half_len))
    arm_b = math_node("MULTIPLY", math_node("LESS_THAN", dy, half_width), math_node("LESS_THAN", dx, half_len))
    cross = math_node("MAXIMUM", arm_a, arm_b)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.006, 0.006, 0.007, 1.0)
    mix.inputs["B"].default_value = (0.55, 0.55, 0.58, 1.0)
    L.new(cross, mix.inputs["Factor"])
    L.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


YAM_SHELL = None
YAM_GRIP = None


def yam_material(body_name: str, geom: dict):
    """The i2rt YAM: white plastic shell, dark gripper (mesh names gripper / tip_left / tip_right)."""
    global YAM_SHELL, YAM_GRIP
    mesh = (geom.get("mesh_name") or "").split("_", 1)[-1]
    if mesh in ("gripper", "tip_left", "tip_right"):
        if YAM_GRIP is None:
            # the rig's grippers are black; mid-grey fingers vanished against the white hose
            YAM_GRIP = bs.make_principled_material("mat__yam_grip", base_color=(0.012, 0.012, 0.014, 1.0),
                                                   roughness=0.42, specular=0.3, noise_bump=0.2, noise_scale=500.0)
        return YAM_GRIP
    if YAM_SHELL is None:
        YAM_SHELL = bs.make_principled_material("mat__yam_shell", base_color=(0.88, 0.89, 0.91, 1.0),
                                                roughness=0.32, specular=0.55, noise_bump=0.14,
                                                noise_scale=420.0, roughness_variation=0.1)
    return YAM_SHELL


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------

class Scene:
    def __init__(self, bundle: Path, tl: dict, arrays, base_blend: Path | None, peg_labels: bool = False):
        self.tl, self.arrays = tl, arrays
        self.overhead = False
        if base_blend is not None:
            bpy.ops.wm.open_mainfile(filepath=str(base_blend.resolve()))
        else:
            bs.reset_scene()
        scn = bpy.context.scene
        keep_look = base_blend is not None and any(o.type == "LIGHT" for o in scn.objects)
        table_z = float(tl["table_z"])

        # the arms: the render repo's body/geom pass, with the YAM materials
        robot = json.loads((bundle / "robot_scene.json").read_text())
        bsp.material_for_geom = yam_material
        self.bodies = bsp.build_bodies_and_geoms(robot, floor_expand=1.0)
        self.body_names = list(tl["body_names"])

        # floor, world, lights -- unless the base scene brings its own
        if not keep_look:
            self._studio(table_z)
        elif "floor" not in bpy.data.objects:
            self._floor(table_z)

        # pegs, labels, goal-side markers
        peg_mat = bs.make_principled_material("mat__peg", base_color=linear(PEG_RGB) + (1.0,), roughness=0.35)
        self.cam = self._camera()
        label_mat = object_colour_material("mat__label")
        for i, p in enumerate(tl["pegs"]):
            bpy.ops.mesh.primitive_cylinder_add(radius=p["r"], depth=p["h"], vertices=48,
                                                location=(p["x"], p["y"], table_z + p["h"] / 2))
            peg = bpy.context.active_object
            peg.name = "peg%d" % i
            peg.data.materials.append(peg_mat)
            bpy.ops.object.shade_smooth()
            if not peg_labels:
                continue
            txt = bpy.data.curves.new("peg_label%d" % i, "FONT")
            txt.body = "p%d" % i
            txt.size = 0.03
            txt.align_x = "CENTER"
            txt.font = bpy.data.fonts.load(FONT_BOLD, check_existing=True)
            lab = bpy.data.objects.new("peg_label%d" % i, txt)
            scn.collection.objects.link(lab)
            lab.location = (p["x"], p["y"], table_z + p["h"] + 0.022)
            con = lab.constraints.new("TRACK_TO")
            con.target = self.cam
            con.track_axis = "TRACK_Z"
            con.up_axis = "UP_Y"
            txt.materials.append(label_mat)
            lab.color = linear((0.95, 0.95, 0.97)) + (1.0,)
            lab["emit"] = 2.0
        marker_mat = object_colour_material("mat__side_marker", roughness=0.4)
        self.markers = []
        for lm in tl["landmarks"]:
            p = tl["pegs"][lm["peg"]]
            n = np.array(lm["normal"] + [0.0])
            c = np.array([p["x"], p["y"], table_z + 0.0015])
            tip = np.array(lm["point"][:2] + [table_z + 0.0015])
            gate = new_curve("gate%d" % lm["peg"], np.stack([c + n * (p["r"] + 0.004), tip - n * 0.012]),
                             0.0016, marker_mat, resolution=2)
            bpy.ops.mesh.primitive_torus_add(major_radius=0.012, minor_radius=0.0022, location=tuple(tip))
            ring = bpy.context.active_object
            ring.name = "side%d" % lm["peg"]
            ring.data.materials.append(marker_mat)
            self.markers.append((lm["peg"], gate, ring))

        # hose and the MPPI drawables
        hose0 = hose_polyline(arrays["hose"][0])
        length = float(np.linalg.norm(np.diff(hose0, axis=0), axis=1).sum())
        self.hose = new_curve("hose", catmull_rom(hose0, 4), float(tl["hose_radius"]), hose_material(length),
                              resolution=6)
        self.draw = []
        mats = {}
        for d in tl["drawables"]:
            group = d["group"]
            if group not in mats:
                mats[group] = object_colour_material("mat__%s" % group, roughness=0.3 if group != "selected_forecast"
                                                     else 0.5)
            pts = np.asarray(d["points"], float)
            if group in ("forecast", "selected_forecast"):
                pts = catmull_rom(hose_polyline(pts), 3)
            obj = new_curve("draw%04d_%s" % (d["id"], group), pts, d["radius"], mats[group],
                            resolution=4 if d["radius"] > 0.01 else 2)
            obj.hide_render = True
            obj["emit"] = float(d["emission"])
            self.draw.append((obj, d, linear(d["rgba"][:3])))

        self.ghosts = self._ghosts(bundle, robot) if tl.get("ghosts") else []
        scn.render.fps = int(tl["fps"])
        scn.frame_start, scn.frame_end = 0, int(tl["n_frames"]) - 1

    def _ghosts(self, bundle: Path, robot: dict) -> list:
        """Translucent copies of the arm that would move, one per ghost in the timeline (the chair clip's
        ghost samples). Their poses come from ghost_tracks.npz (the rig's IK of each candidate's plan); a ghost
        without a track -- the IK refused it -- is not drawn."""
        path = bundle / "ghost_tracks.npz"
        if not path.exists():
            logger.warning("timeline has ghosts but %s is missing: run mppi_viz_robot.py --ghosts", path)
            return []
        G = np.load(path)
        names = {t: [str(n) for n in G["body_names_" + t]] for t in ("L", "R")}
        geoms = {b["name"]: b["geoms"] for b in robot["bodies"]}
        mat = object_colour_material("mat__ghost_arm", roughness=0.4)
        meshes, out = {}, []
        for g in self.tl["ghosts"]:
            for t in ("L", "R"):
                key = "%d/%s/" % (g["id"], t)
                if key + "pos" not in G.files:
                    continue
                empties, parts = [], []
                for bname in names[t]:
                    e = bpy.data.objects.new("ghost%03d_%s" % (g["id"], bname), None)
                    bpy.context.scene.collection.objects.link(e)
                    e.rotation_mode = "QUATERNION"
                    for geom in geoms.get(bname, []):
                        f = geom.get("mesh_file")
                        if not f:
                            continue
                        if f not in meshes:
                            meshes[f] = bs.import_mesh(Path(f)).copy()
                            meshes[f].materials.clear()
                            meshes[f].materials.append(mat)
                        o = bpy.data.objects.new("ghost%03d_%s_%s" % (g["id"], bname, geom["name"]), meshes[f])
                        bpy.context.scene.collection.objects.link(o)
                        o.parent = e
                        o.location = Vector(geom["local_pos"])
                        o.rotation_mode = "QUATERNION"
                        o.rotation_quaternion = bs.quat_wxyz(geom["local_quat"])
                        o.visible_shadow = False
                        o.hide_render = True
                        o["emit"] = float(g["emission"])
                        parts.append(o)
                    empties.append(e)
                out.append(dict(g=g, empties=empties, parts=parts, pos=G[key + "pos"], quat=G[key + "quat"],
                                rgb=linear(g["rgba"][:3])))
        logger.info("ghost arms: %d arm copies for %d ghosts", len(out), len(self.tl["ghosts"]))
        return out

    def _floor(self, table_z):
        bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, table_z))
        floor = bpy.context.active_object
        floor.name = "floor"
        floor.data.materials.append(floor_material())

    def _studio(self, table_z):
        scn = bpy.context.scene
        world = bpy.data.worlds.new("World")
        scn.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes["Background"]
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 0.0
        self._floor(table_z)

        def area(name, loc, target, energy, size, colour=(1.0, 1.0, 1.0)):
            light = bpy.data.lights.new(name, "AREA")
            light.energy, light.size, light.color = energy, size, colour
            obj = bpy.data.objects.new(name, light)
            scn.collection.objects.link(obj)
            obj.location = loc
            direction = Vector(target) - Vector(loc)
            obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

        # key behind and above the camera, so the glossy floor never mirrors it into view; a low rim
        # from the far side gives the floor its fading sheen (the reference render's horizon glow)
        area("key", (-0.95, -0.55, 2.9), (0.5, -0.1, 0.78), 480.0, 1.4)
        area("rim", (2.2, 0.35, 1.05), (0.5, -0.1, 0.8), 70.0, 1.5, (0.90, 0.94, 1.0))
        area("fill", (-0.2, 1.2, 1.5), (0.5, -0.1, 0.8), 40.0, 1.0, (1.0, 0.97, 0.92))

    def _camera(self):
        data = bpy.data.cameras.new("mppi_cam")
        data.lens = float(self.tl.get("lens_mm", 34.0))
        data.clip_start = 0.02
        cam = bpy.data.objects.new("mppi_cam", data)
        bpy.context.scene.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        return cam

    # ------------------------------------------------------------------ per frame
    def apply(self, f: int) -> None:
        tl, A = self.tl, self.arrays
        t = f / float(tl["fps"])
        for bi, name in enumerate(self.body_names):
            obj = self.bodies.get(name)
            if obj is None:
                continue
            obj.location = Vector(A["body_pos"][f, bi].tolist())
            obj.rotation_quaternion = bs.quat_wxyz(A["body_quat"][f, bi])
        set_curve_points(self.hose, catmull_rom(hose_polyline(A["hose"][f]), 4))
        for peg, gate, ring in self.markers:
            s = float(A["pegs"][f, peg])
            rgb = linear(tuple(BAD_RGB[i] + s * (OK_RGB[i] - BAD_RGB[i]) for i in range(3)))
            for obj in (gate, ring):
                obj.color = rgb + (1.0,)
                obj["emit"] = 2.0
        for obj, d, rgb in self.draw:
            a = interp_keys(d["alpha"], t, 1.0) * d["rgba"][3]
            g = interp_keys(d["grow"], t, 1.0)
            hide = a < 0.004 or g < 0.002
            if obj.hide_render != hide:
                obj.hide_render = hide
            if hide:
                continue
            obj.color = rgb + (a,)
            if abs(obj.data.bevel_factor_end - g) > 1e-4:
                obj.data.bevel_factor_end = g
        for gh in self.ghosts:
            g = gh["g"]
            a = interp_keys(g["alpha"], t, 0.0) * g["rgba"][3]
            hide = a < 0.004
            for o in gh["parts"]:
                if o.hide_render != hide:
                    o.hide_render = hide
            if hide:
                continue
            x = interp_keys(g["motion"], t, 0.0) * (len(gh["pos"]) - 1)
            i = min(int(x), len(gh["pos"]) - 2)
            w = x - i
            for bi, e in enumerate(gh["empties"]):
                e.location = Vector(((1 - w) * gh["pos"][i, bi] + w * gh["pos"][i + 1, bi]).tolist())
                q0, q1 = gh["quat"][i, bi], gh["quat"][i + 1, bi]
                q = (1 - w) * q0 + w * (q1 if float(np.dot(q0, q1)) >= 0 else -q1)
                e.rotation_quaternion = bs.quat_wxyz(q / np.linalg.norm(q))
            for o in gh["parts"]:
                o.color = gh["rgb"] + (a,)
        if self.overhead:
            return
        eye, target = Vector(A["cam_eye"][f].tolist()), Vector(A["cam_target"][f].tolist())
        self.cam.location = eye
        self.cam.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()

    def use_overhead_camera(self) -> tuple[int, int]:
        """Pose the camera as the rig's overhead camera (OpenCV K and T_world_overhead). -> (w, h)

        The whole-chain check: arms from IK, hose from the observations and the film offset all have to
        land on the pixels the real camera recorded, or something upstream is wrong.
        """
        from mathutils import Matrix
        ov = self.tl["overhead"]
        K, T = np.asarray(ov["K"], float), np.asarray(ov["T_world_overhead"], float)
        w, h = ov["size"]
        M = np.eye(4)
        M[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])     # OpenCV (+z fwd, +y down) -> Blender camera
        M[:3, 3] = T[:3, 3]
        self.cam.matrix_world = Matrix(M.tolist())
        cam = self.cam.data
        cam.sensor_fit = "HORIZONTAL"
        cam.sensor_width = 36.0
        cam.lens = K[0, 0] / w * 36.0
        cam.shift_x = -(K[0, 2] - w / 2.0) / w
        cam.shift_y = (K[1, 2] - h / 2.0) / w
        self.overhead = True
        return int(w), int(h)


def configure(res, samples: int) -> None:
    bs.configure_cycles(samples=samples, res_x=res[0], res_y=res[1])
    scn = bpy.context.scene
    scn.cycles.transparent_max_bounces = 64
    scn.cycles.max_bounces = 6
    scn.cycles.caustics_reflective = False
    scn.cycles.caustics_refractive = False
    scn.render.use_persistent_data = True
    scn.view_settings.exposure = 0.0
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.compression = 30


# ---------------------------------------------------------------------------
# side-by-side composition
# ---------------------------------------------------------------------------

class Composer:
    """[Blender render + its legend | the real film, untouched]. Pure PIL; reads the timeline, never the run.

    ``annotate`` adds back the captions (cycle / stage / detail), the run clock and pause badge on the film, and
    the bottom strip with peg sides and the run's timeline bar.
    """

    def __init__(self, tl: dict, arrays, panel, annotate: bool = False):
        from PIL import ImageFont
        self.tl, self.A = tl, arrays
        self.pw, self.ph = panel
        self.annotate = annotate
        self.sh = int(self.ph * 0.19) if annotate else 0
        self.W, self.H = 2 * self.pw, self.ph + self.sh
        k = self.ph / 720.0
        self.k = k
        self.f_title = ImageFont.truetype(FONT_BOLD, int(28 * k))
        self.f_stage = ImageFont.truetype(FONT, int(20 * k))
        self.f_small = ImageFont.truetype(FONT, int(15 * k))
        self.f_tag = ImageFont.truetype(FONT_BOLD, int(14 * k))
        self.f_detail = ImageFont.truetype(FONT, int(19 * k))
        stops = np.asarray(tl["colorbar"], float)          # cost colour at u = 0 (low) .. 1 (high)
        u = np.linspace(0, 1, 256)
        grid = np.linspace(0, 1, len(stops))
        self.bar = np.stack([np.interp(u, grid, stops[:, c]) for c in range(3)], 1).__mul__(255).astype(np.uint8)
        self.total = float(tl["duration_s"])

    def _shadow_text(self, draw, xy, text, font, fill):
        x, y = xy
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=fill)

    def frame(self, f: int, blender_png: Path):
        from PIL import Image, ImageDraw
        tl, A, k = self.tl, self.A, self.k
        cap = tl["captions"][int(A["caption"][f])]
        canvas = Image.new("RGB", (self.W, self.H), (8, 8, 10))
        left = Image.open(blender_png).convert("RGB")
        if left.size != (self.pw, self.ph):
            left = left.resize((self.pw, self.ph), Image.LANCZOS)
        canvas.paste(left, (0, 0))
        rgb = Image.open(tl["rgb"][f]).convert("RGB").resize((self.pw, self.ph), Image.LANCZOS)
        if not self.annotate:
            canvas.paste(rgb, (self.pw, 0))
            if cap.get("legend") == "mppi":            # the only text: the cost colour bar, while it applies
                self._cost_bar(ImageDraw.Draw(canvas, "RGBA"), int(22 * k))
            return canvas
        paused = float(A["speed"][f]) == 0.0
        if paused:
            rgb = Image.blend(rgb, Image.new("RGB", rgb.size, (0, 0, 0)), 0.35)
        canvas.paste(rgb, (self.pw, 0))
        d = ImageDraw.Draw(canvas, "RGBA")
        pad = int(22 * k)

        # Blender panel: title / stage top-left, legend bottom-left (the reference's caption spot)
        d.rectangle([0, 0, self.pw, int(86 * k)], fill=(0, 0, 0, 110))
        self._shadow_text(d, (pad, int(14 * k)), cap.get("title", ""), self.f_title, (245, 245, 247))
        self._shadow_text(d, (pad, int(52 * k)), cap.get("stage", ""), self.f_stage, (200, 202, 208))
        d.text((self.pw - pad, int(14 * k)), "RECONSTRUCTED FROM THE RUN LOG", font=self.f_tag,
               fill=(150, 150, 158), anchor="ra")
        self._legend(d, cap, pad)

        # film panel: what it is, its clock, its speed
        x0 = self.pw + pad
        d.rectangle([self.pw, 0, self.W, int(46 * k)], fill=(0, 0, 0, 120))
        d.text((x0, int(14 * k)), "REAL RIG  ·  overhead camera", font=self.f_tag, fill=(235, 235, 238))
        rt = float(A["run_time"][f])
        clock = "run time %d:%04.1f" % (int(rt // 60), rt % 60)
        if paused:
            ps = cap.get("planning_s")
            badge = "PAUSED · planning took %.1f s" % ps if (cap.get("mode") == "plan" and ps) else "PAUSED"
            colour = (255, 196, 64)
        else:
            badge = "%g× real time" % float(A["speed"][f])
            colour = (120, 220, 170)
        d.text((self.W - pad, int(14 * k)), "%s   %s" % (clock, badge), font=self.f_tag, fill=colour, anchor="ra")

        # strip: detail, peg sides, the run's timeline with a playhead
        y0 = self.ph
        d.rectangle([0, y0, self.W, self.H], fill=(14, 14, 17))
        d.text((pad, y0 + int(16 * k)), cap.get("detail", ""), font=self.f_detail, fill=(220, 220, 226))
        sides = A["pegs"][f]
        xs = self.W - pad - int(40 * k) * len(sides)
        d.text((xs - int(12 * k), y0 + int(18 * k)), "peg sides", font=self.f_small, fill=(170, 170, 178), anchor="ra")
        for i, s in enumerate(sides):
            c = tuple(int(255 * (BAD_RGB[j] + float(s) * (OK_RGB[j] - BAD_RGB[j]))) for j in range(3))
            cx = xs + int(40 * k) * i + int(14 * k)
            cy = y0 + int(28 * k)
            r = int(13 * k)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
            d.text((cx, cy), "p%d" % i, font=self.f_tag, fill=(10, 10, 12), anchor="mm")
        self._timeline(d, f, y0 + int(64 * k), pad)
        return canvas

    def _cost_bar(self, d, pad):
        """Lower-left: the candidates' colour scale and nothing else."""
        k = self.k
        bar_w, bar_h = int(220 * k), int(10 * k)
        top = self.ph - pad - int(46 * k)
        d.rounded_rectangle([pad - int(10 * k), top - int(10 * k), pad + bar_w + int(10 * k), self.ph - pad + int(4 * k)],
                            radius=int(8 * k), fill=(0, 0, 0, 150))
        self._shadow_text(d, (pad, top), "MPPI cost", self.f_small, (225, 225, 230))
        yb = top + int(22 * k)
        for i in range(bar_w):
            c = tuple(int(v) for v in self.bar[int(i * (len(self.bar) - 1) / max(1, bar_w - 1))])
            d.line([pad + i, yb, pad + i, yb + bar_h], fill=c)
        d.text((pad, yb + bar_h + int(3 * k)), "low", font=self.f_small, fill=tuple(int(v) for v in self.bar[0]))
        d.text((pad + bar_w, yb + bar_h + int(3 * k)), "high", font=self.f_small,
               fill=tuple(int(v) for v in self.bar[-1]), anchor="ra")

    FALLBACK_LINES = dict(probe=[[(.45, .80, .98), "blue = the commanded probe move"],
                                 [(.96, .96, .98), "white = the path the rig's gripper drove"]])

    def _legend(self, d, cap, pad):
        """Lower-left, on a dark backing so an arm passing behind never eats the text. The lines come from the
        timeline caption (it knows what is drawn); 'mppi' adds the cost colour bar above them."""
        k = self.k
        mode = cap.get("legend")
        if not mode:
            return
        lines = [(tuple(int(255 * c) for c in rgb), text)
                 for rgb, text in cap.get("legend_lines", self.FALLBACK_LINES.get(mode, []))]
        bar = mode == "mppi"
        title = "%s colour = %s (this cycle)" % (cap.get("subject", "sample"), cap.get("cost_word", "MPPI cost"))
        line_h, bar_h, bar_w = int(22 * k), int(10 * k), int(220 * k)
        texts = [t for _, t in lines] + ([title] if bar else [])
        if not texts:
            return
        width = max([d.textlength(t, font=self.f_small) for t in texts] + [bar_w if bar else 0])
        height = len(lines) * line_h + (int(64 * k) if bar else 0)
        top = self.ph - pad - height
        d.rounded_rectangle([pad - int(10 * k), top - int(10 * k), pad + width + int(12 * k), self.ph - pad + int(4 * k)],
                            radius=int(8 * k), fill=(0, 0, 0, 150))
        y = top
        if bar:
            lo, hi = cap.get("cost_range", [0.0, 1.0])
            self._shadow_text(d, (pad, y), title, self.f_small, (225, 225, 230))
            yb = y + line_h
            for i in range(bar_w):
                c = tuple(int(v) for v in self.bar[int(i * (len(self.bar) - 1) / max(1, bar_w - 1))])
                d.line([pad + i, yb, pad + i, yb + bar_h], fill=c)
            good, bad = tuple(int(v) for v in self.bar[0]), tuple(int(v) for v in self.bar[-1])
            d.text((pad, yb + bar_h + int(4 * k)), "good  %.2f" % lo, font=self.f_small, fill=good)
            d.text((pad + bar_w, yb + bar_h + int(4 * k)), "%.2f  bad" % hi, font=self.f_small, fill=bad, anchor="ra")
            y += int(64 * k)
        for colour, text in lines:
            self._shadow_text(d, (pad, y), text, self.f_small, colour)
            y += line_h

    def _timeline(self, d, f, y, pad):
        k = self.k
        x0, x1 = pad, self.W - pad
        h = int(12 * k)
        t = f / float(self.tl["fps"])
        scale = (x1 - x0) / self.total
        labels_done = set()
        for s in self.tl["segments"]:
            a, b = x0 + s["t0"] * scale, x0 + s["t1"] * scale
            base = SEGMENT_RGB.get(s["kind"], (80, 80, 80))
            active = s["t0"] <= t < s["t1"] or (s is self.tl["segments"][-1] and t >= s["t0"])
            colour = base if active or t >= s["t1"] else tuple(int(c * 0.35) for c in base)
            d.rectangle([a + 1, y, b - 1, y + h], fill=colour)
            label = None
            if s["kind"] == "probe" and "probes" not in labels_done:
                label, key = "calibration probes", "probes"
            elif s["kind"] == "plan":
                label, key = "cycle %d" % s["cycle"], s["label"]
            if label and key not in labels_done:
                labels_done.add(key)
                d.text((a + 2, y + h + int(5 * k)), label, font=self.f_small, fill=(150, 150, 158))
        px = x0 + t * scale
        d.rectangle([px - 1, y - int(5 * k), px + 1, y + h + int(5 * k)], fill=(255, 255, 255))
        lx = x1
        for kind, name in (("result", "result"), ("exec", "execute"), ("plan", "plan"), ("probe", "probe")):
            w = d.textlength(name, font=self.f_small)
            d.text((lx, y + h + int(5 * k)), name, font=self.f_small, fill=(150, 150, 158), anchor="ra")
            lx -= w + int(10 * k)
            d.rectangle([lx - int(2 * k), y + h + int(9 * k), lx + int(6 * k), y + h + int(17 * k)],
                        fill=SEGMENT_RGB[kind])
            lx -= int(20 * k)


def encode(composer: Composer, frames, render_dir: Path, output: Path, fps: int, crf: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (composer.W, composer.H), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = None
    for f in frames:
        png = render_dir / ("f%05d.png" % f)
        if png.exists():
            last = png
        if last is None:
            continue
        proc.stdin.write(np.asarray(composer.frame(f, last), np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    logger.info("wrote %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_frames(spec: str | None, n: int):
    if not spec:
        return list(range(n))
    a, _, b = spec.partition(":")
    lo = int(a) if a else 0
    hi = int(b) if b else n
    return list(range(max(0, lo), min(n, hi)))


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None, help="rendered Blender frames (default <bundle>/render)")
    p.add_argument("--output", type=Path, default=None, help="side-by-side mp4 (default <bundle>/mppi_run.mp4)")
    p.add_argument("--blender-scene", type=Path, default=None, help="optional base .blend (see module doc)")
    p.add_argument("--panel", default="1280x720", help="size of EACH panel")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--frames", default=None, help="a:b frame range (default all)")
    p.add_argument("--step", type=int, default=1, help="render every n-th frame (the video holds the rest)")
    p.add_argument("--preview", action="store_true", help="half-size panels, 16 samples")
    p.add_argument("--resume", action="store_true", help="skip frames already rendered")
    p.add_argument("--compose-only", action="store_true")
    p.add_argument("--no-compose", action="store_true")
    p.add_argument("--save-blend", type=Path, default=None)
    p.add_argument("--still", type=int, default=None, help="render one frame, compose it to <output>.png, stop")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--peg-labels", action="store_true", help="draw p0/p1/p2 above the pegs (text in the render)")
    p.add_argument("--annotate", action="store_true",
                   help="add captions, the run clock and the bottom timeline strip (default: render + legend | film)")
    p.add_argument("--view", choices=("orbit", "overhead"), default="orbit",
                   help="overhead = the rig's own camera, for the whole-chain check (with --still)")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    bundle = a.bundle.resolve()
    tl = json.loads((bundle / "timeline.json").read_text())
    arrays = dict(np.load(bundle / "timeline.npz"))
    pw, ph = (int(v) for v in a.panel.lower().split("x"))
    samples = a.samples
    if a.preview:
        pw, ph, samples = pw // 2, ph // 2, 16
    out_dir = (a.out_dir or bundle / ("render_preview" if a.preview else "render")).resolve()
    output = (a.output or bundle / ("mppi_run_preview.mp4" if a.preview else "mppi_run.mp4")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(tl["n_frames"])
    frames = parse_frames(a.frames, n)
    if a.still is not None:
        frames = [a.still]
    composer = Composer(tl, arrays, (pw, ph), annotate=a.annotate)

    if not a.compose_only:
        t0 = time.time()
        scene = Scene(bundle, tl, arrays, a.blender_scene, peg_labels=a.peg_labels)
        if a.view == "overhead":
            pw, ph = scene.use_overhead_camera()
            out_dir = out_dir.parent / (out_dir.name + "_overhead")
            out_dir.mkdir(parents=True, exist_ok=True)
        configure((pw, ph), samples)
        logger.info("scene built in %.1f s: %d drawables, %d frames to render", time.time() - t0,
                    len(scene.draw), len(frames[::a.step]))
        if a.save_blend:
            scene.apply(frames[0])
            bpy.ops.wm.save_as_mainfile(filepath=str(a.save_blend.resolve()))
            logger.info("saved %s", a.save_blend)
        scn = bpy.context.scene
        todo = frames[::a.step]
        t_start = time.time()
        for i, f in enumerate(todo):
            png = out_dir / ("f%05d.png" % f)
            if a.resume and png.exists():
                continue
            scn.frame_set(f)
            scene.apply(f)
            scn.render.filepath = str(png)
            bpy.ops.render.render(write_still=True)
            if i % 25 == 0 or i == len(todo) - 1:
                el = time.time() - t_start
                logger.info("frame %d (%d/%d)  %.2f s/frame  eta %.0f min", f, i + 1, len(todo),
                            el / (i + 1), el / (i + 1) * (len(todo) - i - 1) / 60)

    if a.still is not None and a.view == "overhead":
        from PIL import Image
        mine = Image.open(out_dir / ("f%05d.png" % a.still)).convert("RGB")
        real = Image.open(tl["rgb"][a.still]).convert("RGB").resize(mine.size)
        sheet = Image.new("RGB", (mine.size[0] * 3, mine.size[1]))
        for i, im in enumerate((mine, real, Image.blend(mine, real, 0.5))):
            sheet.paste(im, (i * mine.size[0], 0))
        still = output.with_suffix(".overhead.f%05d.png" % a.still)
        sheet.save(still)
        logger.info("wrote %s (render | film | 50/50)", still)
        return
    if a.still is not None:
        img = composer.frame(a.still, out_dir / ("f%05d.png" % a.still))
        still = output.with_suffix(".f%05d.png" % a.still)
        img.save(still)
        logger.info("wrote %s", still)
        return
    if not a.no_compose:
        encode(composer, frames, out_dir, output, int(tl["fps"]), a.crf)


if __name__ == "__main__":
    main()
