"""Replay a graph-as-policy trial through MuJoCo and dump portable scene data.

The eval log captures robot joint trajectories + camera frames but not body
poses. To re-render the trial in Blender we need to know where every body
(robot links, crate, etc.) was at each step. This script reconstructs that
by replaying the joints through the same libero env the eval used, then
extracting ``data.xpos`` / ``data.xquat`` per body per step.

Outputs into ``<out_dir>/``:

* ``traj.npz``
    - ``timestamps``        : (T,) float64
    - ``body_pos``          : (T, NB, 3) float32, world-frame positions
    - ``body_quat``         : (T, NB, 4) float32, world-frame quaternions (wxyz)
    - ``body_names``        : (NB,) <U64
    - ``joint_positions``   : (T, 14) float32 (mirror of input, for sanity)
    - ``grippers``          : (T,) float32
* ``scene.json``
    Per-body / per-geom asset graph (mesh paths, local pose, rgba, material).
* ``cameras.json``
    Camera intrinsics + extrinsics (static + per-step) for every logged camera.
* ``scene_combined.xml`` (best-effort)
    Compiled-from-robosuite MJCF text. Convenience copy; Blender uses
    ``scene.json`` as the source of truth.

This script MUST be run from the graph-as-policy uv env so ``libero`` and
``services.sim_bridge`` are importable. Example::

    cd ~/Projects/graph-as-policy
    uv run python ~/Projects/blender-robot-render/render/replay.py \\
        --trial-dir outputs/libero_crate_washing/eval/task_00/trial_15_rc1_reward1.000_pass \\
        --task-file examples/libero_crate_washing/task.yaml \\
        --out ~/Projects/blender-robot-render/outputs/trial_15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("replay")


# ---------------------------------------------------------------------------
# Trial / task discovery
# ---------------------------------------------------------------------------

_TRIAL_RE = re.compile(r"trial_(\d+)_")


def trial_seed_from_dir(trial_dir: Path) -> int:
    """Extract the integer trial index from ``trial_15_rc1_reward1.000_pass``.

    The eval runner uses ``seed = trial_idx`` when calling ``env.reset(seed=...)``,
    which in turn picks row ``((seed) - 1) % len(init_states)`` from the libero
    init-state table. Reproducing that here is what makes the replay match the
    original trial.
    """
    m = _TRIAL_RE.search(trial_dir.name)
    if not m:
        raise ValueError(f"could not parse trial index from {trial_dir.name}")
    return int(m.group(1))


def load_task_yaml(task_file: Path) -> dict:
    """Load the GaP task.yaml — we only need ``suite_name`` and ``task_id``."""
    import yaml

    with task_file.open() as f:
        data = yaml.safe_load(f)
    return data


def discover_suite_and_task_id(meta: dict, task_yaml: dict | None) -> tuple[str, int]:
    """Pull suite_name + task_id from task.yaml; fall back to meta if needed."""
    if task_yaml is not None:
        suites = task_yaml.get("suites") or []
        if suites:
            s = suites[0]
            return str(s["suite_name"]), int(s.get("task_id", 0))
    # The scene_log meta.json doesn't currently record suite_name, but the
    # MJCF path is a strong hint.
    mjcf_path = meta.get("scene_mjcf", "")
    if "crate_washing" in mjcf_path:
        return "libero_crate_washing", 0
    raise RuntimeError(
        "could not infer suite_name / task_id — pass --task-file explicitly"
    )


# ---------------------------------------------------------------------------
# Robosuite env construction (delegates to graph-as-policy's wrapper)
# ---------------------------------------------------------------------------

def build_env(
    suite_name: str,
    task_id: int,
    max_steps: int = 2000,
    crate_yaw_range_deg: float = 0.0,
    crate_xy_range_m: float = 0.0,
    crate_respawn_enabled: bool = False,
):
    """Construct the same env wrapper sim_bridge uses for eval.

    Keeping it identical (controller, camera names, joint qpos addresses)
    guarantees that re-applying ``positions`` rows from joints.npz produces
    the same body poses as during eval, modulo physics jitter from gripper
    closure / contact.

    ``crate_yaw_range_deg`` and ``crate_xy_range_m`` mirror the per-crate
    randomization knobs in ``examples/libero_crate_washing/task.yaml``. The
    eval's ``reset(seed=trial_idx)`` re-derives the per-crate yaw + xy from
    the same seed, so passing the same ranges + seed here reproduces the
    randomized stack the trial actually used.
    """
    from services.sim_bridge.env.libero_bimanual_env import LiberoBimanualEnv

    env = LiberoBimanualEnv(
        suite_name=suite_name,
        task_id=task_id,
        cam_h=128,
        cam_w=128,
        max_steps=max_steps,
        control_freq=20,
        crate_yaw_range_deg=crate_yaw_range_deg,
        crate_xy_range_m=crate_xy_range_m,
        crate_random_seed=None,  # let reset(seed=N) drive it
        crate_respawn_enabled=crate_respawn_enabled,
    )
    return env


# ---------------------------------------------------------------------------
# MJCF / model introspection
# ---------------------------------------------------------------------------

def _mat_rgba(model, mat_id: int) -> list[float] | None:
    if mat_id < 0 or mat_id >= model.nmat:
        return None
    try:
        return [float(x) for x in model.mat_rgba[mat_id]]
    except Exception:
        return None


def _mesh_file_lookup(model) -> dict[int, str]:
    """Map mesh_id → absolute file path.

    MuJoCo's compiled model doesn't keep the original file path next to each
    mesh — only the vertex/face buffers. Robosuite's MJCF embeds absolute paths
    in ``<asset><mesh file="...">``, so the most reliable way to recover them
    is to parse the source XML (``mj_saveLastXML`` text) ourselves.
    """
    return {}  # populated by parse_source_xml below


def parse_source_xml(xml_text: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Parse the composed MJCF XML and pull out mesh and material defs.

    Returns ``(meshes, materials)`` where:

    * ``meshes[name] = {"file": str, "scale": [3 floats]}``
    * ``materials[name] = {"rgba": [4 floats] | None, "texture": str | None,
                           "specular": float, "shininess": float, "reflectance": float}``

    We tolerate the XML being missing some fields — anything not present comes
    back as ``None`` and the Blender side fills in sensible defaults.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    meshes: dict[str, dict] = {}
    materials: dict[str, dict] = {}

    for asset in root.iter("asset"):
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            if name is None:
                continue
            file = mesh.get("file")
            scale = mesh.get("scale")
            scale_arr = [1.0, 1.0, 1.0]
            if scale:
                parts = [float(s) for s in scale.split()]
                if len(parts) == 3:
                    scale_arr = parts
            meshes[name] = {"file": file, "scale": scale_arr}
        for mat in asset.findall("material"):
            name = mat.get("name")
            if name is None:
                continue
            rgba_s = mat.get("rgba")
            rgba = (
                [float(x) for x in rgba_s.split()]
                if rgba_s
                else None
            )
            materials[name] = {
                "rgba": rgba,
                "texture": mat.get("texture"),
                "specular": float(mat.get("specular", "0.5")),
                "shininess": float(mat.get("shininess", "0.5")),
                "reflectance": float(mat.get("reflectance", "0.0")),
            }
    return meshes, materials


def _quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(a, b):
    """Hamilton product, wxyz convention."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (wxyz)."""
    import numpy as np
    w, x, y, z = q
    # Convert to rotation matrix and multiply
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R @ np.asarray(v, dtype=np.float64)


def build_scene_graph(model, source_xml: str) -> dict:
    """Walk ``mjModel`` and produce a JSON-serializable scene description.

    Only visual geoms (``group <= 2``) are emitted — collision geoms (group 3)
    are dropped to keep the Blender scene clean.

    For mesh geoms we strip MuJoCo's internal ``mesh_quat`` / ``mesh_pos``
    out of the compiled ``geom_quat`` / ``geom_pos``. Those are applied at
    compile time to align the mesh with its inertial frame, and the
    compiled fields are ``user_transform ∘ mesh_transform``. Since Blender
    loads the raw OBJ vertices (no inertial alignment), we want to apply
    only the user-specified transform — typically identity for MJCFs that
    don't override the geom quat. Without this, every mesh is rotated
    twice and ends up sideways relative to its body.
    """
    import mujoco

    meshes_decl, materials_decl = parse_source_xml(source_xml)

    bodies = []
    for bi in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or f"body_{bi}"
        parent = int(model.body_parentid[bi])
        bodies.append({
            "index": bi,
            "name": bname,
            "parent_index": parent,
            "parent_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world",
            "geoms": [],
        })

    skip_visual_groups = {3, 4, 5}  # collision / debug layers
    type_to_str = {
        getattr(mujoco.mjtGeom, name): name.replace("mjGEOM_", "").lower()
        for name in dir(mujoco.mjtGeom)
        if name.startswith("mjGEOM_")
    }

    for gi in range(model.ngeom):
        group = int(model.geom_group[gi])
        if group in skip_visual_groups:
            continue
        gtype = int(model.geom_type[gi])
        gtype_str = type_to_str.get(gtype, str(gtype))
        bi = int(model.geom_bodyid[gi])
        gname = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            or f"geom_{gi}"
        )

        mesh_id = int(model.geom_dataid[gi])
        mesh_name = None
        mesh_file = None
        mesh_scale = [1.0, 1.0, 1.0]
        mesh_quat_internal = (1.0, 0.0, 0.0, 0.0)
        mesh_pos_internal = (0.0, 0.0, 0.0)
        if gtype_str == "mesh" and mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            decl = meshes_decl.get(mesh_name) if mesh_name else None
            if decl is not None:
                mesh_file = decl["file"]
                mesh_scale = list(decl["scale"])
            mesh_quat_internal = tuple(float(x) for x in model.mesh_quat[mesh_id])
            mesh_pos_internal = tuple(float(x) for x in model.mesh_pos[mesh_id])

        mat_id = int(model.geom_matid[gi])
        mat_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id)
            if mat_id >= 0
            else None
        )
        mat_props = materials_decl.get(mat_name) if mat_name else None

        rgba = [float(x) for x in model.geom_rgba[gi]]
        # MuJoCo uses (-1, -1, -1, -1) as "no override; use material"
        if all(r < 0 for r in rgba[:3]):
            rgba = None  # will fall back to material rgba

        # Strip the mesh's internal alignment quat/pos out of the geom's
        # compiled transform. ``geom_pos`` = user_pos + R(user_quat) @ mesh_pos
        # and ``geom_quat`` = user_quat * mesh_quat, so to recover the
        # user-specified transform we right-multiply by mesh_quat^{-1}.
        geom_pos = tuple(float(x) for x in model.geom_pos[gi])
        geom_quat = tuple(float(x) for x in model.geom_quat[gi])
        if gtype_str == "mesh" and mesh_id >= 0:
            user_quat = _quat_mul(geom_quat, _quat_conj(mesh_quat_internal))
            offset_world = _quat_rotate(user_quat, mesh_pos_internal)
            user_pos = (
                geom_pos[0] - float(offset_world[0]),
                geom_pos[1] - float(offset_world[1]),
                geom_pos[2] - float(offset_world[2]),
            )
        else:
            user_pos = geom_pos
            user_quat = geom_quat

        bodies[bi]["geoms"].append({
            "index": gi,
            "name": gname,
            "type": gtype_str,
            "size": [float(s) for s in model.geom_size[gi]],
            "local_pos": list(user_pos),
            "local_quat": list(user_quat),
            "rgba_geom": rgba,
            "material": mat_name,
            "material_props": mat_props,
            "mesh_name": mesh_name,
            "mesh_file": mesh_file,
            "mesh_scale": mesh_scale,
            "group": group,
        })

    cameras = []
    for ci in range(model.ncam):
        cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, ci)
        bi = int(model.cam_bodyid[ci])
        cameras.append({
            "index": ci,
            "name": cname,
            "body_index": bi,
            "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or "world",
            "local_pos": [float(x) for x in model.cam_pos[ci]],
            "local_quat": [float(x) for x in model.cam_quat[ci]],
            "fovy": float(model.cam_fovy[ci]),
        })

    return {
        "bodies": bodies,
        "cameras": cameras,
        "asset_meshes": meshes_decl,
        "asset_materials": materials_decl,
    }


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def replay(
    trial_dir: Path,
    out_dir: Path,
    task_file: Path | None = None,
    snap_logged_robot: bool = False,
    dense_frames: bool = False,
) -> None:
    """Replay one trial and write traj.npz + scene.json into ``out_dir``."""
    import mujoco

    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs --------------------------------------------------------------
    meta_path = trial_dir / "trace" / "scene_log" / "meta.json"
    joints_path = trial_dir / "trace" / "scene_log" / "joints.npz"
    if not meta_path.exists() or not joints_path.exists():
        raise FileNotFoundError(
            f"trial_dir missing meta.json or joints.npz: {trial_dir}"
        )
    meta = json.loads(meta_path.read_text())
    jd = np.load(joints_path, allow_pickle=False)
    positions = jd["positions"]  # (T, 14)
    grippers = jd["grippers"]  # (T,)
    timestamps = jd["timestamps"]  # (T,)
    T = int(positions.shape[0])
    logger.info("loaded %d joint steps from %s", T, joints_path)

    task_yaml = load_task_yaml(task_file) if task_file else None
    suite_name, task_id = discover_suite_and_task_id(meta, task_yaml)
    seed = trial_seed_from_dir(trial_dir)

    # Pull per-crate randomization config from task.yaml so the replayed
    # stack matches the trial's randomized poses (yaw_range_deg, xy_range_m).
    yaw_deg = 0.0
    xy_m = 0.0
    crate_respawn_enabled = False
    if task_yaml is not None:
        env_cfg = (task_yaml.get("environment") or {})
        crate_rand = env_cfg.get("crate_randomization") or {}
        yaw_deg = float(crate_rand.get("yaw_range_deg") or 0.0)
        xy_m = float(crate_rand.get("xy_range_m") or 0.0)
        crate_respawn = env_cfg.get("crate_respawn") or {}
        crate_respawn_enabled = bool(crate_respawn.get("enabled", False))
    logger.info(
        "suite=%s task_id=%d seed=%d crate_yaw_deg=±%.1f "
        "crate_xy_m=±%.3f crate_respawn=%s",
        suite_name, task_id, seed, yaw_deg, xy_m, crate_respawn_enabled,
    )

    # Build env ----------------------------------------------------------
    env = build_env(
        suite_name=suite_name,
        task_id=task_id,
        crate_yaw_range_deg=yaw_deg,
        crate_xy_range_m=xy_m,
        crate_respawn_enabled=crate_respawn_enabled,
    )
    # The eval used horizon=2000 (≈100 s of sim @ control_freq=20). Our
    # converge-to-target inner loop multiplies effective step count, so we
    # routinely cross the horizon mid-replay. Disabling the horizon-based
    # done check is the right move for offline replay since we just want
    # to keep collecting body poses until the recorded trajectory ends.
    env.handle_env.env.ignore_done = True
    env.handle_env.env.done = False
    env.reset(seed=seed)
    env.handle_env.env.ignore_done = True  # reset() resets it
    env.handle_env.env.done = False

    sim = env.handle_env.env.sim
    # mujoco_py-style binding; the underlying mujoco.MjModel/MjData are at
    # ``sim.model._model`` / ``sim.data._data`` (robosuite wraps them).
    raw_model = sim.model._model if hasattr(sim.model, "_model") else sim.model
    raw_data = sim.data._data if hasattr(sim.data, "_data") else sim.data

    # Snapshot the source XML for material/mesh extraction. ``mj_saveLastXML``
    # writes the *last loaded* MJCF, which for robosuite is the composed temp
    # file (scene + robots + grippers).
    xml_path = out_dir / "scene_combined.xml"
    try:
        mujoco.mj_saveLastXML(str(xml_path), raw_model)
        source_xml = xml_path.read_text()
        logger.info("saved combined MJCF (%d bytes) → %s", len(source_xml), xml_path)
    except Exception:
        logger.warning("mj_saveLastXML failed; falling back to scene meta MJCF only", exc_info=True)
        source_xml = Path(meta["scene_mjcf"]).read_text()
        xml_path.write_text(source_xml)

    # Asset graph (bodies + geoms + cameras) -----------------------------
    scene_graph = build_scene_graph(raw_model, source_xml)
    body_names = [b["name"] for b in scene_graph["bodies"]]
    NB = len(body_names)

    video_states_path = trial_dir / "trace" / "scene_log" / "video_sim_states.npz"
    if video_states_path.exists():
        vs = np.load(video_states_path, allow_pickle=False)
        logged_body_names = [str(name) for name in vs["body_names"]]
        logged_index = {name: i for i, name in enumerate(logged_body_names)}
        missing = [name for name in body_names if name not in logged_index]
        if missing:
            raise ValueError(
                "video_sim_states.npz does not contain scene bodies: "
                + ", ".join(missing[:10])
            )
        body_idx = np.array([logged_index[name] for name in body_names], dtype=np.int64)
        body_pos = vs["body_pos"][:, body_idx, :].astype(np.float32)
        body_quat = vs["body_quat"][:, body_idx, :].astype(np.float32)
        timestamps = vs["timestamps"].astype(np.float64)
        if len(timestamps) > 1 and not np.any(np.diff(timestamps) > 0):
            timestamps = vs["frame_indices"].astype(np.float64) / 20.0
        qpos = vs["qpos"].astype(np.float32)

        np.savez_compressed(
            out_dir / "traj.npz",
            timestamps=timestamps,
            body_pos=body_pos,
            body_quat=body_quat,
            body_names=np.array(body_names),
            joint_positions=qpos,
            grippers=np.zeros((body_pos.shape[0],), dtype=np.float32),
        )
        logger.info(
            "wrote %s from exact video sim states (%d frames)",
            out_dir / "traj.npz",
            body_pos.shape[0],
        )

        (out_dir / "scene.json").write_text(json.dumps(scene_graph, indent=2))
        logger.info("wrote %s", out_dir / "scene.json")

        cam_log_dir = trial_dir / "trace" / "scene_log" / "cameras"
        cameras_out = {}
        for cam in scene_graph["cameras"]:
            cname = cam["name"]
            entry: dict = {"static": cam}
            log_dir = cam_log_dir / cname if cname else None
            if log_dir and log_dir.exists():
                intr = log_dir / "intrinsics.npy"
                poses = log_dir / "poses.npy"
                ts = log_dir / "timestamps.npy"
                if intr.exists():
                    entry["intrinsics"] = np.load(intr).tolist()
                if poses.exists():
                    entry["poses"] = np.load(poses).tolist()
                if ts.exists():
                    entry["timestamps"] = np.load(ts).tolist()
            cameras_out[cname] = entry
        (out_dir / "cameras.json").write_text(json.dumps(cameras_out, indent=2))
        logger.info("wrote %s", out_dir / "cameras.json")
        return

    # Joint qpos addresses for the two arms (mirror libero_bimanual_env)
    def _first(x):
        return x[0] if isinstance(x, tuple) else x

    left_addr = [
        int(_first(sim.model.get_joint_qpos_addr(f"robot0_joint{i}")))
        for i in range(1, 8)
    ]
    right_addr = [
        int(_first(sim.model.get_joint_qpos_addr(f"robot1_joint{i}")))
        for i in range(1, 8)
    ]
    left_gripper_addr = _gripper_qpos_addrs(sim.model, 0)
    right_gripper_addr = _gripper_qpos_addrs(sim.model, 1)

    # Pre-allocate output buffers for waypoint replay. Dense replay appends
    # every MuJoCo control step instead so Blender does not have to bridge
    # multi-second gaps between sparse logged waypoints.
    body_pos = np.zeros((T, NB, 3), dtype=np.float32)
    body_quat = np.zeros((T, NB, 4), dtype=np.float32)
    dense_body_pos: list[np.ndarray] = []
    dense_body_quat: list[np.ndarray] = []
    dense_timestamps: list[float] = []
    dense_positions: list[np.ndarray] = []
    dense_grippers: list[float] = []
    dense_dt = 1.0 / float(getattr(env, "_control_freq", 20) or 20)
    dense_t0 = float(timestamps[0]) if len(timestamps) else 0.0

    # Physics replay using the same env wrapper the eval used. The action
    # format is *delta* joint-velocity scaled by control_freq — not absolute
    # target — and the wrapper already has a converge-toward-target helper
    # (``move_to_joints_blocking_both``) that does the right thing. We drive
    # both arms toward each recorded waypoint, set the gripper fraction
    # symmetrically, then snapshot ``xpos``/``xquat`` for every body. Contact
    # forces during convergence let the crate's free joint evolve correctly.
    #
    # ``grippers[t]`` in joints.npz is *already* a fraction in [0, 1]
    # (1 = open, 0 = closed), normalised by sim_bridge's
    # ``_log_joints_from_obs`` callsite. Pass it through directly.

    arm_dof = 7
    converge_tol = 0.02      # rad; loose enough that one recorded waypoint
    converge_max_steps = 12  # ~0.6 s at control_freq=20 — short enough not to
                             # double the original eval runtime, long enough
                             # for typical inter-waypoint distance.

    import time as _time
    t0 = _time.time()
    crate11_id = _body_id_of(raw_model, "crate_box_11")
    crate11_qpos_start = int(getattr(env, "_crate11_qpos_start", -1))
    last_convey_t = -1
    last_convey_qpos: np.ndarray | None = None
    last_convey_dense_t = -1
    last_convey_dense_qpos: np.ndarray | None = None

    def _crate_in_convey_band(qpos7: np.ndarray) -> bool:
        xyz = np.asarray(qpos7[:3], dtype=np.float64)
        return bool(
            0.88 <= xyz[2] <= 1.12
            and xyz[0] <= 1.25
            and abs(xyz[1]) <= 0.50
        )

    inner_env = env.handle_env.env  # robosuite base env — has the .done flag
    old_record_frame = getattr(env, "_record_frame", None)
    old_record_frames = getattr(env, "_record_frames", False)
    old_subsample_rate = getattr(env, "_subsample_rate", None)

    def _current_joint_row() -> np.ndarray:
        return np.concatenate(
            [
                np.array([sim.data.qpos[a] for a in left_addr], dtype=np.float32),
                np.array([sim.data.qpos[a] for a in right_addr], dtype=np.float32),
            ]
        )

    def _capture_dense_frame() -> None:
        nonlocal last_convey_dense_t, last_convey_dense_qpos
        dense_body_pos.append(raw_data.xpos.astype(np.float32).copy())
        dense_body_quat.append(raw_data.xquat.astype(np.float32).copy())
        dense_timestamps.append(dense_t0 + dense_dt * len(dense_body_pos))
        dense_positions.append(_current_joint_row())
        dense_grippers.append(
            float((env._gripper_fraction[0] + env._gripper_fraction[1]) / 2.0)
        )
        if crate11_qpos_start >= 0:
            qpos7 = np.array(
                sim.data.qpos[crate11_qpos_start : crate11_qpos_start + 7],
                dtype=np.float64,
            )
            if _crate_in_convey_band(qpos7):
                last_convey_dense_t = len(dense_body_pos) - 1
                last_convey_dense_qpos = qpos7.copy()

    if dense_frames:
        _capture_dense_frame()
        env._record_frame = _capture_dense_frame  # type: ignore[attr-defined]
        env._record_frames = True  # type: ignore[attr-defined]
        env._subsample_rate = 1  # type: ignore[attr-defined]

    last_valid_t = -1
    try:
        for t in range(T):
            # The episode flips ``done=True`` once the BDDL goal condition is
            # met (crate lifted past threshold). robosuite blocks further
            # ``step()`` calls; we just clear the flag so we can keep
            # collecting body poses for the post-success motion (gripper
            # release, arm retract, conveyor animation).
            if getattr(inner_env, "done", False):
                inner_env.done = False
            target_l = positions[t, :arm_dof].astype(np.float64)
            target_r = positions[t, arm_dof:2 * arm_dof].astype(np.float64)
            frac = float(np.clip(float(grippers[t]), 0.0, 1.0))
            env._set_gripper(frac, arm_id=0)
            env._set_gripper(frac, arm_id=1)
            try:
                env.move_to_joints_blocking_both(
                    target_l, target_r,
                    tolerance=converge_tol,
                    max_steps=converge_max_steps,
                )
            except ValueError as e:
                logger.warning("step %d: %s; truncating replay here", t, e)
                break
            if snap_logged_robot:
                # Optional visual-only mode: preserve replayed object physics, then
                # put the robot back on the exact joint state logged by the trial.
                sim.data.qpos[left_addr] = target_l
                sim.data.qpos[right_addr] = target_r
                _set_gripper_qpos(sim.data.qpos, left_gripper_addr, frac)
                _set_gripper_qpos(sim.data.qpos, right_gripper_addr, frac)
                sim.forward()
            if not dense_frames:
                body_pos[t] = raw_data.xpos.astype(np.float32)
                body_quat[t] = raw_data.xquat.astype(np.float32)
            if crate11_qpos_start >= 0:
                qpos7 = np.array(
                    sim.data.qpos[crate11_qpos_start : crate11_qpos_start + 7],
                    dtype=np.float64,
                )
                if _crate_in_convey_band(qpos7):
                    last_convey_t = t
                    last_convey_qpos = qpos7.copy()
            last_valid_t = t
            if t % 25 == 0 or t == T - 1:
                logger.info(
                    "  step %3d/%d  crate11.z=%.3f  grip=%.2f  (%.1fs elapsed)",
                    t + 1, T,
                    float(raw_data.xpos[crate11_id, 2]),
                    frac,
                    _time.time() - t0,
                )
    finally:
        if dense_frames:
            env._record_frames = old_record_frames  # type: ignore[attr-defined]
            if old_record_frame is not None:
                env._record_frame = old_record_frame  # type: ignore[attr-defined]
            if old_subsample_rate is not None:
                env._subsample_rate = old_subsample_rate  # type: ignore[attr-defined]

    if last_valid_t < 0:
        raise RuntimeError("replay produced no valid frames")
    T_out = len(dense_body_pos) if dense_frames else last_valid_t + 1
    if dense_frames:
        body_pos = np.stack(dense_body_pos, axis=0)
        body_quat = np.stack(dense_body_quat, axis=0)
        timestamps = np.asarray(dense_timestamps, dtype=np.float64)
        positions = np.stack(dense_positions, axis=0).astype(np.float32)
        grippers = np.asarray(dense_grippers, dtype=np.float32)
        logger.info("dense replay captured %d MuJoCo control frames", T_out)
    elif T_out < T:
        logger.info("trimming output to %d valid frames (of %d recorded)", T_out, T)
        body_pos = body_pos[:T_out]
        body_quat = body_quat[:T_out]
        timestamps = timestamps[:T_out]
        positions = positions[:T_out]
        grippers = grippers[:T_out]

    if crate_respawn_enabled and hasattr(env, "respawn_top_crate"):
        if crate11_qpos_start >= 0:
            current_qpos = np.array(
                sim.data.qpos[crate11_qpos_start : crate11_qpos_start + 7],
                dtype=np.float64,
            )
            restore_qpos = (
                last_convey_dense_qpos if dense_frames else last_convey_qpos
            )
            restore_t = last_convey_dense_t if dense_frames else last_convey_t
            if (
                not _crate_in_convey_band(current_qpos)
                and restore_qpos is not None
                and restore_t >= 0
            ):
                keep = restore_t + 1
                logger.info(
                    "restoring last placed crate pose from replay frame %d "
                    "before conveyor animation",
                    keep,
                )
                body_pos = body_pos[:keep]
                body_quat = body_quat[:keep]
                timestamps = timestamps[:keep]
                positions = positions[:keep]
                grippers = grippers[:keep]
                T_out = keep
                sim.data.qpos[
                    crate11_qpos_start : crate11_qpos_start + 7
                ] = restore_qpos
                try:
                    free_jid = int(sim.model.joint_name2id("crate_box_11_free"))
                    vel_start = int(sim.model.jnt_dofadr[free_jid])
                    sim.data.qvel[vel_start : vel_start + 6] = 0.0
                except Exception:
                    pass
                sim.forward()

        extra_pos: list[np.ndarray] = []
        extra_quat: list[np.ndarray] = []

        old_record_frame = getattr(env, "_record_frame", None)
        old_record_frames = getattr(env, "_record_frames", False)
        old_subsample_rate = getattr(env, "_subsample_rate", None)

        def _capture_respawn_frame() -> None:
            extra_pos.append(raw_data.xpos.astype(np.float32).copy())
            extra_quat.append(raw_data.xquat.astype(np.float32).copy())

        env._record_frame = _capture_respawn_frame  # type: ignore[attr-defined]
        env._record_frames = True  # type: ignore[attr-defined]
        env._subsample_rate = 1  # type: ignore[attr-defined]
        try:
            env.respawn_top_crate(rng_seed=seed)
        finally:
            env._record_frames = old_record_frames  # type: ignore[attr-defined]
            if old_record_frame is not None:
                env._record_frame = old_record_frame  # type: ignore[attr-defined]
            if old_subsample_rate is not None:
                env._subsample_rate = old_subsample_rate  # type: ignore[attr-defined]

        if extra_pos:
            n_extra = len(extra_pos)
            logger.info(
                "appending %d respawn/conveyor frames after recorded trajectory",
                n_extra,
            )
            body_pos = np.concatenate(
                [body_pos, np.stack(extra_pos, axis=0)], axis=0
            )
            body_quat = np.concatenate(
                [body_quat, np.stack(extra_quat, axis=0)], axis=0
            )
            dt = 1.0 / float(getattr(env, "_control_freq", 20) or 20)
            start_ts = float(timestamps[-1]) if len(timestamps) else 0.0
            extra_ts = start_ts + dt * np.arange(1, n_extra + 1, dtype=np.float64)
            timestamps = np.concatenate([timestamps, extra_ts], axis=0)
            if len(positions):
                positions = np.concatenate(
                    [positions, np.repeat(positions[-1:], n_extra, axis=0)],
                    axis=0,
                )
            if len(grippers):
                grippers = np.concatenate(
                    [grippers, np.repeat(grippers[-1:], n_extra, axis=0)],
                    axis=0,
                )
            T_out = int(body_pos.shape[0])
        else:
            logger.info(
                "crate_respawn enabled, but no conveyor frames were produced "
                "(crate may not be in the placed band)"
            )

    logger.info("replayed %d steps; %d bodies; %.1fs total",
                T_out, NB, _time.time() - t0)

    # Save traj.npz ------------------------------------------------------
    np.savez_compressed(
        out_dir / "traj.npz",
        timestamps=timestamps.astype(np.float64),
        body_pos=body_pos,
        body_quat=body_quat,
        body_names=np.array(body_names),
        joint_positions=positions.astype(np.float32),
        grippers=grippers.astype(np.float32),
    )
    logger.info("wrote %s", out_dir / "traj.npz")

    # Save scene.json ----------------------------------------------------
    (out_dir / "scene.json").write_text(json.dumps(scene_graph, indent=2))
    logger.info("wrote %s", out_dir / "scene.json")

    # Save cameras.json --------------------------------------------------
    cam_log_dir = trial_dir / "trace" / "scene_log" / "cameras"
    cameras_out = {}
    for cam in scene_graph["cameras"]:
        cname = cam["name"]
        entry: dict = {"static": cam}
        log_dir = cam_log_dir / cname if cname else None
        if log_dir and log_dir.exists():
            intr = log_dir / "intrinsics.npy"
            poses = log_dir / "poses.npy"
            ts = log_dir / "timestamps.npy"
            if intr.exists():
                entry["intrinsics"] = np.load(intr).tolist()
            if poses.exists():
                entry["poses"] = np.load(poses).tolist()
            if ts.exists():
                entry["timestamps"] = np.load(ts).tolist()
        cameras_out[cname] = entry
    (out_dir / "cameras.json").write_text(json.dumps(cameras_out, indent=2))
    logger.info("wrote %s", out_dir / "cameras.json")


def _body_id_of(model, name: str) -> int:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return int(bid) if bid >= 0 else 0


def _gripper_qpos_addrs(model_wrapped, arm_idx: int) -> list[int]:
    """Find gripper joint qpos addrs for one arm — best-effort.

    Robosuite names gripper joints variably (panda: ``gripper{0,1}_finger_joint{1,2}``,
    others differ). We just probe a few common names.
    """
    candidates = [
        f"gripper{arm_idx}_finger_joint1",
        f"gripper{arm_idx}_finger_joint2",
        f"gripper{arm_idx}_finger1_joint",
        f"gripper{arm_idx}_finger2_joint",
    ]
    addrs = []
    for n in candidates:
        try:
            a = model_wrapped.get_joint_qpos_addr(n)
            if isinstance(a, tuple):
                a = a[0]
            addrs.append(int(a))
        except Exception:
            pass
    return addrs


def _set_gripper_qpos(qpos, addrs: list[int], fraction: float) -> None:
    """Set gripper joints from a [0, 1] open fraction.

    1.0 = fully open (positive finger qpos for Panda), 0.0 = closed. We use a
    simple linear ramp; this is just for visualization, not physics.
    """
    if not addrs:
        return
    # Panda finger joint range ≈ [0, 0.04]
    open_q = 0.04 * float(np.clip(fraction, 0.0, 1.0))
    for a in addrs:
        qpos[a] = open_q


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial-dir", required=True, type=Path)
    p.add_argument("--task-file", type=Path, default=None,
                   help="GaP task.yaml — optional, used to disambiguate suite_name")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--snap-logged-robot",
        action="store_true",
        help=(
            "After physics replay, snap robot joints to the logged values before "
            "writing body poses. This improves robot-video alignment but can make "
            "robot/object contact less physically consistent."
        ),
    )
    p.add_argument(
        "--dense-frames",
        action="store_true",
        help=(
            "Write every MuJoCo control step from the replay instead of only one "
            "frame per logged waypoint. This avoids visible pose jumps when the "
            "trial joint log is sparse."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Make sure ``services`` is importable when invoked from outside the
    # graph-as-policy tree.
    repo_root = Path(__file__).resolve()
    while repo_root != repo_root.parent:
        if (repo_root / "vos").is_dir() and (repo_root / "services").is_dir():
            sys.path.insert(0, str(repo_root))
            break
        repo_root = repo_root.parent
    # Fallback: the user's machine
    gap = Path.home() / "Projects" / "graph-as-policy"
    if gap.is_dir() and str(gap) not in sys.path:
        sys.path.insert(0, str(gap))

    replay(
        args.trial_dir.resolve(),
        args.out.resolve(),
        args.task_file,
        snap_logged_robot=args.snap_logged_robot,
        dense_frames=args.dense_frames,
    )


if __name__ == "__main__":
    main()
