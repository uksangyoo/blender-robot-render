"""A white, publication-quality two-arm YAM scene: table slab, pegs, hose capsules, both arms posed.

Read-only. Compiles in memory (MjSpec), renders offscreen with EGL, moves nothing: no CAN bus is opened
and the i2rt driver is never imported. It must run with the arm stack's interpreter, from its directory:

    cd ~/Projects/hose-routing/mild-trackdlo/yam_bimanual
    .venv_trace/bin/python ~/Projects/hose-routing/scripts/v5/render_showcase_robot.py --run <run_dir>

WHY NOT the overhead photograph. Drawing the solved arms into the ZED frame (`crate_flipping`
`render_grasp.py`) is the better WHOLE-CHAIN check and should still be used for that. This exists for a
different purpose -- a figure a reader can understand -- so the scene is synthetic, on white, lit.

The colours match `showcase_style.py` so a render and a plot can sit in the same figure.

Every non-obvious MuJoCo fact below was measured on this machine (mujoco 3.3.4, EGL, RTX 5090), and the
comments say what goes wrong without it. The whole-chain check that keeps it honest is
`render_showcase_robot.py --verify`: it reads `grasp_site` out of the composed world model and compares it
with the position the phase JSON commanded (worst 0.48 mm over the 70 keyframes of this run).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

TABLE_Z = 0.75
JAW_STROKE = 0.0475          # joint7/joint8 slide range; qpos 0 = SHUT, 0.0475 = OPEN (measured, see pose)

# showcase_style.py, as MuJoCo rgba: PEG #E3A12C, PRED #7C3AED, a slate hose, a light-grey arm.
PEG_RGBA = '0.890 0.631 0.173 1'
GHOST_RGBA = '0.486 0.227 0.929 1'
HOSE_RGBA = '0.169 0.200 0.247 1'
LINK_RGBA = (0.815, 0.822, 0.835, 1.0)
GRIP_RGBA = (0.255, 0.278, 0.320, 1.0)

# Camera presets, chosen by rendering the sweep in `render_showcase_robot.py --sweep`.
# (lookat, distance, azimuth, elevation). A free camera's field of view is the XML's
# <visual><global fovy>, not a property of MjvCamera.
VIEWS = {
    'hero': ([.45, -.06, .80], 1.52, 208, -35),
    'wide': ([.45, -.06, .80], 1.70, 231, -30),
    'mirror': ([.45, -.06, .80], 1.60, 152, -31),
    'plan': ([.45, -.11, .78], 1.62, 180, -89.9),
}


def arm_mjcf_path():
    """The i2rt-generated single-arm MJCF. Mesh paths inside are ABSOLUTE, so the
    file can be re-parsed from anywhere and MjSpec.from_file needs no asset dir."""
    from i2rt.robots.utils import (ArmType, GripperType,
                                   combine_arm_and_gripper_xml)
    return combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)


def wxyz(R):
    x, y, z, w = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return (w, x, y, z)


def world_xml(pegs, hose_polyline, hose_radius=0.035, width=2400, height=1500,
              fovy=32.0, table_z=TABLE_Z, offsamples=8, shadowsize=16384,
              table=(0.425, 0.0, 0.29, 0.58), hose_rgba=HOSE_RGBA,
              peg_rgba=PEG_RGBA, reflectance=0.0,
              tubes=(), extra_cameras=""):
    """Parent scene. `table` is (cx, cy, half_x, half_y) of the slab."""
    cx, cy, hx, hy = table
    pg = "\n".join(
        '    <geom name="peg%d" type="cylinder" material="peg" '
        'pos="%.6f %.6f %.6f" size="%.4f %.4f"/>'
        % (i, p["x"], p["y"], table_z + p["height"] / 2.0,
           p["radius"], p["height"] / 2.0)
        for i, p in enumerate(pegs))
    P = np.asarray(hose_polyline, float)
    cap = "\n".join(
        '    <geom name="hose%02d" type="capsule" material="hose" size="%.5f" '
        'fromto="%.6f %.6f %.6f %.6f %.6f %.6f"/>'
        % (i, hose_radius, *P[i], *P[i + 1]) for i in range(len(P) - 1))
    tubes = [dict(t) for t in tubes]
    gh = "\n".join(
        '    <geom name="tube%03d_%02d" type="capsule" material="tube%03d" size="%.5f" '
        'fromto="%.6f %.6f %.6f %.6f %.6f %.6f" contype="0" conaffinity="0" group="3"/>'
        % (k, i, k, float(t.get('radius', .009)),
           *np.asarray(t['points'], float)[i], *np.asarray(t['points'], float)[i + 1])
        for k, t in enumerate(tubes)
        for i in range(len(np.asarray(t['points'])) - 1))
    tube_materials = "\n".join(
        '    <material name="tube%03d" rgba="%s" specular="0.10" shininess="0.10"/>'
        % (k, t.get('rgba', GHOST_RGBA)) for k, t in enumerate(tubes))
    return f"""
<mujoco model="hose_routing_scene">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" fovy="{fovy}"/>
    <quality shadowsize="{shadowsize}" offsamples="{offsamples}"
             numslices="80" numstacks="40" numquads="8"/>
    <map shadowclip="1.2" shadowscale="0.9" znear="0.01" zfar="30"
         fogstart="10" fogend="20"/>
    <headlight ambient="0.30 0.30 0.32" diffuse="0.10 0.10 0.11"
               specular="0.03 0.03 0.03" active="1"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="flat" width="128" height="768"
             rgb1="1 1 1" rgb2="1 1 1"/>
    <material name="table" rgba="0.880 0.882 0.888 1" specular="0.12"
              shininess="0.06" reflectance="{reflectance}"/>
    <material name="peg"   rgba="{peg_rgba}" specular="0.30" shininess="0.30"/>
    <material name="hose"  rgba="{hose_rgba}" specular="0.28" shininess="0.26"/>
{tube_materials}
  </asset>
  <worldbody>
    <light name="key"  pos="-0.5 -1.3 2.7" dir="0.34 0.60 -1" directional="true"
           castshadow="true"  diffuse="0.52 0.52 0.52" specular="0.14 0.14 0.14"/>
    <light name="fill" pos=" 1.9  1.5 2.3" dir="-0.62 -0.47 -1" directional="true"
           castshadow="false" diffuse="0.18 0.18 0.20" specular="0.02 0.02 0.02"/>
    <light name="rim"  pos=" 1.7 -1.7 1.5" dir="-0.62  0.62 -0.6" directional="true"
           castshadow="false" diffuse="0.10 0.10 0.12" specular="0.0 0.0 0.0"/>
    <geom name="table" type="box" material="table"
          pos="{cx:.4f} {cy:.4f} {table_z - 0.012:.4f}" size="{hx:.4f} {hy:.4f} 0.012"/>
{pg}
{cap}
{gh}
{extra_cameras}  </worldbody>
</mujoco>
""".strip()


def build(pegs, hose_polyline, T_world_left, T_world_right,
          link_rgba=LINK_RGBA, grip_rgba=GRIP_RGBA, **kw):
    """-> (model, data, qadr{L,R}->6 addrs, gadr{L,R}->2 addrs)"""
    spec = mujoco.MjSpec.from_string(world_xml(pegs, hose_polyline, **kw))
    arm_path = arm_mjcf_path()
    for tag, T in (("L", T_world_left), ("R", T_world_right)):
        child = mujoco.MjSpec.from_file(arm_path)
        # Recolour IN THE CHILD: attach() prefixes MATERIAL names too, so a geom
        # pointing at the parent's "arm" would be rewritten to "L_arm" and fall
        # back to a default material. Define them here and they follow the rename.
        child.add_material(name="arm", rgba=list(link_rgba),
                           specular=0.42, shininess=0.42)
        child.add_material(name="grip", rgba=list(grip_rgba),
                           specular=0.55, shininess=0.50)
        # The arm's geoms are UNNAMED in the i2rt MJCF (g.name == ""); the mesh
        # name is the only handle on which link a geom belongs to.
        for g in child.geoms:
            g.material = ("grip" if g.meshname in ("gripper", "tip_left", "tip_right")
                          else "arm")
            # MuJoCo rule: a geom rgba that DIFFERS from the internal default
            # (0.5 0.5 0.5 1) overrides the material colour. The i2rt MJCF gives
            # every link its own garish rgba, so it must be reset to exactly the
            # default or the material is ignored and the arm renders in its
            # factory colours (or, if set to white, blown out white).
            g.rgba = [0.5, 0.5, 0.5, 1.0]
        T = np.asarray(T, float).reshape(4, 4)
        f = spec.worldbody.add_frame(pos=T[:3, 3], quat=wxyz(T[:3, :3]))
        spec.attach(child, prefix=tag + "_", frame=f)
    model = spec.compile()
    data = mujoco.MjData(model)
    qadr = {t: [model.joint(f"{t}_joint{i}").qposadr[0] for i in range(1, 7)]
            for t in ("L", "R")}
    gadr = {t: [model.joint(f"{t}_joint{i}").qposadr[0] for i in (7, 8)]
            for t in ("L", "R")}
    return model, data, qadr, gadr


def pose(model, data, qadr, gadr, q_by_tag, grip_by_tag):
    """grip is the PLAN's convention: 1.0 = open, 0.0 = shut (i2rt "0 close 1 open").
    The MJCF slide runs the SAME way: measured on the composed model (2026-09-17), qpos 0
    puts the two pad meshes' centroids 10 mm apart (shut, pads interpenetrating) and
    qpos 0.0475 puts them 103 mm apart (open), so qpos = grip * JAW_STROKE. An earlier
    (1 - grip) here drew every gripper open when the rig's was shut and vice versa.
    On the rig a grip closed on the 70 mm hose reads back ~0.6, not 0: the hose stops the jaws."""
    data.qpos[:] = 0.0
    for tag, q6 in q_by_tag.items():
        for a, v in zip(qadr[tag], np.asarray(q6, float)):
            data.qpos[a] = v
        for a in gadr[tag]:
            data.qpos[a] = float(grip_by_tag[tag]) * JAW_STROKE
    mujoco.mj_forward(model, data)


def view(model, name='hero', **over):
    """One of the named presets, with any field overridden. -> MjvCamera"""
    lookat, distance, azimuth, elevation = VIEWS[name]
    return free_cam(model, over.get('lookat', lookat), over.get('distance', distance),
                    over.get('azimuth', azimuth), over.get('elevation', elevation))


def polyline(centres):
    """(N,3) node centres -> (N+1,3) capsule joints, so a prediction can be drawn like the hose.

    The tracked observation carries its own `pre_polyline` (29 points for 28 capsules); a model
    prediction is 28 centres, and this is the conversion the scene needs to make it a tube.
    """
    c = np.asarray(centres, float)
    mid = .5 * (c[:-1] + c[1:])
    return np.vstack([c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])])


def free_cam(model, lookat, distance, azimuth, elevation):
    """A free camera. NOTE: the field of view is model.vis.global_.fovy (the XML's
    <visual><global fovy>), NOT a property of MjvCamera."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def opts():
    """mjtVisFlag: which GEOMETRY enters the scene."""
    o = mujoco.MjvOption()
    o.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
    o.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = True
    o.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = True
    o.frame = mujoco.mjtFrame.mjFRAME_NONE
    o.label = mujoco.mjtLabel.mjLABEL_NONE
    # The i2rt gripper carries grasp_site (green) and tcp_site (red), both in
    # group 0, and MuJoCo DRAWS sites by default -- a lime dot appears on the
    # hose in every keyframe until this is cleared.
    o.sitegroup[:] = 0
    return o


def render_flags(renderer, shadow=True, reflection=True, skybox=True, haze=False):
    """mjtRndFlag lives on renderer.scene.flags, NOT on MjvOption.
    (mujoco.mjtVisFlag has no mjVIS_SHADOW -- that attribute error is the tell.)
    update_scene() does not reset these, so set them once after constructing."""
    f = renderer.scene.flags
    f[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(shadow)
    f[mujoco.mjtRndFlag.mjRND_REFLECTION] = bool(reflection)
    f[mujoco.mjtRndFlag.mjRND_SKYBOX] = bool(skybox)
    f[mujoco.mjtRndFlag.mjRND_HAZE] = bool(haze)
    f[mujoco.mjtRndFlag.mjRND_SEGMENT] = False
    f[mujoco.mjtRndFlag.mjRND_IDCOLOR] = False
    f[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False


GHOST_GROUP = 3
"""The overlay tubes (a forecast, or a fan of candidate forecasts) sit in geom group 3, which
MjvOption leaves OFF. That is
what makes the two-pass composite below possible -- and it is the ONLY way to get
a clean translucent overlay. Giving the ghost material alpha<1 instead makes
MuJoCo alpha-blend each of the 28 capsules separately, and because consecutive
capsules overlap at their spherical caps the tube comes out covered in scales."""


def opaque_tubes(renderer, data, cam, o):
    """The group-3 tubes drawn solid: what a fan of thin candidate forecasts wants."""
    g = o.geomgroup[GHOST_GROUP]
    o.geomgroup[GHOST_GROUP] = 1
    renderer.update_scene(data, camera=cam, scene_option=o)
    img = renderer.render().copy()
    o.geomgroup[GHOST_GROUP] = g
    return img


def composite_ghost(renderer, data, cam, o, alpha=0.38):
    """-> RGB with the group-3 ghost blended over the scene at a UNIFORM alpha.

    Pass A: ghost hidden -> the picture. Pass B: ghost shown, opaque -> its
    colour, correctly occluded by whatever is in front of it. A segmentation
    pass says which pixels the ghost actually won. Blending only those is
    order-free and seamless.
    """
    import numpy as _np
    g = o.geomgroup[GHOST_GROUP]
    o.geomgroup[GHOST_GROUP] = 0
    renderer.update_scene(data, camera=cam, scene_option=o)
    base = renderer.render().copy()
    o.geomgroup[GHOST_GROUP] = 1
    renderer.update_scene(data, camera=cam, scene_option=o)
    over = renderer.render().copy()
    mask = None
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam, scene_option=o)
        seg = renderer.render()                  # (H, W, 2) int32: objid, objtype
        m = renderer.model
        ids = {i for i in range(m.ngeom) if m.geom(i).name.startswith("tube")}
        mask = _np.isin(seg[:, :, 0], _np.array(sorted(ids), _np.int32))
        mask &= seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    except IndexError:
        # mujoco's renderer maps segmentation ids through an array sized from scene.ngeom, and a
        # scene with ~1400 geoms (a fan of 48 predictions is 48 x 28 capsules) overruns it. The
        # pixels the overlay won are exactly the ones that CHANGED between the two passes, which
        # needs no ids at all.
        mask = None
    finally:
        renderer.disable_segmentation_rendering()
        o.geomgroup[GHOST_GROUP] = g
    if mask is None:
        mask = (base.astype(_np.int16) - over.astype(_np.int16) != 0).any(axis=-1)
    # Blend only the masked PIXELS. Doing it full-frame in float32 costs ~120 ms
    # at 3000x1875 for nothing: 96% of the frame is untouched.
    out = base
    a = float(alpha)
    out[mask] = (base[mask] * (1.0 - a) + over[mask] * a).astype(_np.uint8)
    return out, mask
