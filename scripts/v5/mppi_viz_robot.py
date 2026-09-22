#!/usr/bin/env python3
"""Both YAM arms through every executed phase of a real run, as Blender-ready body tracks. Read-only.

Stage 1 of the Blender MPPI video (`render_mppi_run.py` runs all three). Needs the arm stack's
interpreter, from its own directory, exactly like `render_showcase_robot.py`:

    cd ~/Projects/hose-routing/mild-trackdlo/yam_bimanual
    MUJOCO_GL=egl .venv_trace/bin/python ../../scripts/v5/mppi_viz_robot.py \
        --run ../../outputs/mppi_real_v5/run_20260917_150620 --out <bundle dir>

For every phase the rig executed (6 probes, the slipped probe's retry, 4 cycles) it

  1. solves the rig's own IK for the phase JSON's keyframes (`render_showcase_robot.solve_phase`, the call
     `execute_phase_yam` makes; no CAN bus is opened, the i2rt driver is never imported),
  2. rebuilds the TIME SCHEDULE the held session drove them on (`yam_session.run_phase` staging +
     `execute_phase_yam.drive_lockstep` step timing, with the session's own --seg-time/--first-time/
     --close-time/--home-time read back from run.log), where `move_joints` interpolates the 7 joints
     linearly over each step, and the retreat homes right then left,
  3. samples forward kinematics of the composed two-arm MuJoCo world (`showcase_scene.build`) at 60 Hz.

The jaws follow the gripper READBACK the runner logged after every keyframe (`metrics[...]['grip']`,
[keyframe, role, value]), not the command: shut on the 70 mm hose the rig reads ~0.6, and a command of 0 would
drive the rendered pads through the hose. A keyframe without a readback falls back to its commanded grip.

Writes into --out:
  robot_scene.json   the arm bodies and their visual geoms (mesh file, local pose), the render repo's
                     `replay_peg_climb.build_scene_graph` format
  robot_tracks.npz   per phase tag: t (K,), q (K,2,7) [left,right], body_pos (K,B,3), body_quat (K,B,4 wxyz),
                     tool (K,2,3) grasp_site in world
  robot_schedule.json per phase: every commanded move (arm, keyframe, t0, t1, grip), the motion window,
                     the mover, and the whole-chain check (grasp_site vs commanded, mm)

TIME ZERO of a phase is the first commanded move. Where that sits inside the phase's film is NOT logged;
`mppi_viz_data.py` fits it against the film's own motion.

ASSUMPTION, stated because it is not in the record: each arm connected, and therefore retreats to, joint
zero (the folded pose the film's first frame shows). `yam_session.starts_ref` is not written to disk.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

from mppi_paths import HOSE_ROUTING_ROOT, RENDER_REPO, SCRIPTS

YAM = os.path.expanduser(os.environ.get('YAM_BIMANUAL_DIR') or str(HOSE_ROUTING_ROOT / 'mild-trackdlo/yam_bimanual'))
HR = str(HOSE_ROUTING_ROOT)
RENDER_REPO = str(RENDER_REPO)
for path in (YAM, os.path.join(YAM, 'scripts'), os.path.join(HR, 'scripts'),
             HR, RENDER_REPO, str(SCRIPTS)):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco                                                                 # noqa: E402
import showcase_scene as SC                                                   # noqa: E402
import render_showcase_robot as RS                                            # noqa: E402
import execute_phase_yam as E                                                 # noqa: E402
from render.replay_peg_climb import build_scene_graph                         # noqa: E402

ARMS = ('left', 'right')
TAG = {'left': 'L', 'right': 'R'}
HOME_Q = np.zeros(6)
RATE_HZ = 60.
# yam_session.py defaults, used only when run.log does not carry the session's command line
TIMING_DEFAULT = dict(seg=1.2, first=2.4, close=2.0, home=2.0)


def session_timing(run) -> dict:
    """The held session's step times, from the command line run.log printed when it started."""
    out = dict(TIMING_DEFAULT)
    log = os.path.join(run, 'run.log')
    if os.path.exists(log):
        text = open(log, errors='replace').read()
        for key, flag in (('seg', 'seg-time'), ('first', 'first-time'), ('close', 'close-time'),
                          ('home', 'home-time')):
            m = re.search(r'--%s[ =]([0-9.]+)' % flag, text)
            if m:
                out[key] = float(m.group(1))
    return out


def executed_phases(run, metrics, phases_dir=None) -> list[dict]:
    """Every phase the rig ran, in order, with the tag its film and observation use.

    The earlier campaign (`run_real_mppi.py`) recorded no `plan_file`: its phases are numbered in the order
    the executor wrote them, probes first, so the fallback below is that order. It is not a guess that goes
    unchecked -- `mppi_viz_data.LegacyRigPlanner.verify` rebuilds each phase from the action the record keeps
    beside it, and a wrong pairing shows up there as keyframes that do not match.
    """
    out = []
    for row in metrics.get('probes') or []:
        k, attempt = int(row.get('k', 0)), int(row.get('attempt', 0) or 0)
        tag = 'probe_%03d' % k + ('_retry%d' % attempt if attempt else '')
        out.append(dict(tag=tag, kind='probe', n=k, attempt=attempt, plan=row.get('plan_file'),
                        grip=row.get('grip') or []))
    for row in metrics.get('cycles') or []:
        n = int(row['cycle'])
        out.append(dict(tag='cycle_%03d' % n, kind='cycle', n=n, attempt=0, plan=row.get('plan_file'),
                        grip=row.get('grip') or []))
    rebuilt = json.load(open(os.path.join(phases_dir, 'index.json'))) if phases_dir else None
    legacy = None
    for p in out:
        if rebuilt is not None:              # a run that sent no phases to a rig; they were rebuilt for it
            p['plan'] = rebuilt.get(p['tag'], '')
            continue
        if p['plan']:
            p['plan'] = os.path.join(run, 'phases', os.path.basename(str(p['plan'])))
            continue
        if legacy is None:                    # the earlier campaign: pair phase files with actions, once
            import showcase_data_legacy as DL
            legacy = {k: str(v) for k, v in DL.phase_files(run, metrics).items()}
        p['plan'] = legacy.get(p['tag'], '')
    return [p for p in out if p['plan'] and os.path.exists(p['plan'])]


def schedule(phase, sol, timing, readback=()) -> list[dict]:
    """The commanded moves, timed as the held session drove them. -> [dict(arm, name, q, grip, t0, t1)]

    Mirrors `yam_session.run_phase`: under `meta.sync` the contact stage is one lockstep for both arms,
    otherwise anchor then mover, each alone; then the motion and release stages lockstep with the mover
    commanded first; then the retreat home, right arm then left. Inside a step `drive_lockstep` gives a
    `close` keyframe close_time, the phase's very first move first_time, everything else seg_time; a
    synced step moves both arms at once (threads), a sequential one moves them one after the other.
    """
    meta = phase['meta']
    if meta.get('sweep_deg'):
        raise NotImplementedError('base-sweep phases are not reproduced here')
    rig = meta['rig_arm']
    mover = phase.get('mover') or 'minus_y'
    anchor = 'plus_y' if mover == 'minus_y' else 'minus_y'
    sync = bool(meta.get('sync', False))
    reads = {}                                     # (keyframe, role) -> readbacks in order
    for name, role, value in readback:
        reads.setdefault((str(name), str(role)), []).append(float(value))
    wps = {arm: [] for arm in ARMS}
    for role, s in sol.items():
        wps[s['rig_arm']] = [dict(name=n, q6=np.asarray(q, float),
                                  grip=reads[(n, role)].pop(0) if reads.get((n, role)) else float(g),
                                  commanded_grip=float(g))
                             for n, q, g in zip(s['names'], s['q6'], s['grip'])]
    stages = {arm: E.split_stages(wps[arm]) for arm in ARMS}
    moves, clock = [], [0.]

    def lockstep(lists, first, order=None, synced=True):
        tags = [t for t in (order or sorted(lists)) if t in lists]
        n = max((len(lists[t]) for t in tags), default=0)
        for i in range(n):
            step = [(t, lists[t][i]) for t in tags if i < len(lists[t])]
            if not step:
                continue
            dt = timing['close'] if any(w['name'] == 'close' for _, w in step) else \
                (timing['first'] if first else timing['seg'])
            if synced and len(step) > 1:
                for t, w in step:
                    moves.append(dict(arm=t, name=w['name'], q=w['q6'], grip=w['grip'],
                                      t0=clock[0], t1=clock[0] + dt, stage=E.stage_of(w['name'])))
                clock[0] += dt
            else:
                for t, w in step:
                    moves.append(dict(arm=t, name=w['name'], q=w['q6'], grip=w['grip'],
                                      t0=clock[0], t1=clock[0] + dt, stage=E.stage_of(w['name'])))
                    clock[0] += dt
            first = False
        return first

    first = True
    if sync:
        first = lockstep({t: stages[t][0] for t in ARMS}, first, synced=True)
    else:
        for role in (anchor, mover):
            t = rig[role]
            if wps[t]:
                first = lockstep({t: stages[t][0]}, first, synced=False)
    for si in (1, 2):
        first = lockstep({t: stages[t][si] for t in ARMS}, first, order=(rig[mover], rig[anchor]), synced=sync)
    for t in ('right', 'left'):                     # yam_session.retreat: right first, one after the other
        moves.append(dict(arm=t, name='home', q=HOME_Q.copy(), grip=1., t0=clock[0],
                          t1=clock[0] + timing['home'], stage=3))
        clock[0] += timing['home']
    return moves


def joint_track(moves, times) -> np.ndarray:
    """(K,) times -> (K, 2, 7) joints [left, right], 6 arm joints + grip, linear inside each move."""
    out = np.zeros((len(times), 2, 7))
    for a, arm in enumerate(ARMS):
        seq = [m for m in moves if m['arm'] == arm]
        start = np.append(HOME_Q, 1.)
        for k, t in enumerate(times):
            q = start.copy()
            for m in seq:
                target = np.append(m['q'], m['grip'])
                if t >= m['t1']:
                    q = target
                elif t > m['t0']:
                    u = (t - m['t0']) / max(m['t1'] - m['t0'], 1e-9)
                    q = q + u * (target - q)
                    break
                else:
                    break
            out[k, a] = q
    return out


def mesh_file_map(model) -> dict:
    """{mesh_id: (name, abs path)} for the composed world: attach() prefixed the i2rt names with L_/R_."""
    arm = mujoco.MjSpec.from_file(SC.arm_mjcf_path())
    files = {m.name: m.file for m in arm.meshes}
    out = {}
    for mid in range(model.nmesh):
        name = model.mesh(mid).name
        bare = name.split('_', 1)[1] if name[:2] in ('L_', 'R_') else name
        if bare in files:
            out[mid] = (name, os.path.abspath(files[bare]))
    return out


GHOST_GRIP_SHUT = 0.6     # what the rig's gripper reads back shut on the 70 mm hose (metrics grip readback)


def ghost_tracks(requests, config, out, per_leg=8):
    """Candidate plans -> the moving arm's body poses from grasp through the carry, for the ghost arms.

    Each request is a phase dict in phases/*.json form (`mppi_viz_data.RigPlanner`), solved with the same
    `solve_phase` as the executed phases. A ghost is the arm whose motion-stage keyframes travel (>10 mm), from
    its `close` keyframe to its last motion keyframe, joint-linear between keyframes as `move_joints` drives them,
    `per_leg` samples a leg. A plan with any keyframe the IK rejects there is dropped: the rig would have refused
    it. Cached on a hash of the requests.
    """
    import hashlib
    key = hashlib.sha1(json.dumps(requests, sort_keys=True).encode()).hexdigest()
    path = os.path.join(out, 'ghost_tracks.npz')
    if os.path.exists(path):
        with np.load(path) as z:
            if 'hash' in z.files and str(z['hash']) == key:
                print('ghost tracks up to date (%d requests)' % len(requests))
                return
    model = None
    arrays, kept, dropped = {}, [], []
    for req in requests:
        sol, meta, world = RS.solve_phase(req['phase'], config)
        if model is None:
            hose = np.array([[.45, -.3, .785], [.45, .3, .785]])
            model, data, qadr, gadr = SC.build([], hose, world['left'], world['right'])
            bodies = {t: [i for i in range(model.nbody) if model.body(i).name.startswith(t + '_')
                          and model.body(i).name != t + '_base'] for t in ('L', 'R')}
            for t in ('L', 'R'):
                arrays['body_names_' + t] = np.array([model.body(i).name for i in bodies[t]])
        arms = []
        for role, s in sol.items():
            names = list(s['names'])
            motion = [i for i, n in enumerate(names) if E.stage_of(n) == 1]
            if not motion:
                continue
            start = names.index('close') if 'close' in names else (names.index('grasp') if 'grasp' in names else 0)
            idx = list(range(start, motion[-1] + 1))
            travel = np.linalg.norm(s['commanded'][idx] - s['commanded'][start], axis=1).max()
            if travel <= .01:
                continue
            if not all(s['ok'][i] for i in idx):
                arms = None
                break
            arms.append((s['tag'], s['rig_arm'], s['q6'][idx], np.where(s['grip'][idx] < .5, GHOST_GRIP_SHUT, 1.)))
        if not arms:
            dropped.append(req['id'])
            continue
        for tag, arm, q, g in arms:
            poses_p, poses_q = [], []
            for leg in range(len(q) - 1):
                for u in np.linspace(0, 1, per_leg, endpoint=False):
                    SC.pose(model, data, qadr, gadr, {tag: q[leg] + u * (q[leg + 1] - q[leg])},
                            {tag: g[leg] + u * (g[leg + 1] - g[leg])})
                    poses_p.append(data.xpos[bodies[tag]].copy())
                    poses_q.append(data.xquat[bodies[tag]].copy())
            SC.pose(model, data, qadr, gadr, {tag: q[-1]}, {tag: g[-1]})
            poses_p.append(data.xpos[bodies[tag]].copy())
            poses_q.append(data.xquat[bodies[tag]].copy())
            arrays['%d/%s/pos' % (req['id'], tag)] = np.asarray(poses_p, np.float32)
            arrays['%d/%s/quat' % (req['id'], tag)] = np.asarray(poses_q, np.float32)
        kept.append(req['id'])
    arrays['hash'] = np.array(key)
    arrays['kept'] = np.array(kept, int)
    arrays['dropped'] = np.array(dropped, int)
    np.savez_compressed(path, **arrays)
    print('ghost arms: %d kept, %d dropped (IK refused a motion keyframe) -> %s' % (len(kept), len(dropped), path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True, help='bundle directory')
    ap.add_argument('--config', default=os.path.join(YAM, 'config.yaml'))
    ap.add_argument('--rate', type=float, default=RATE_HZ)
    ap.add_argument('--only', nargs='*', default=None, help='phase tags to export (default all)')
    ap.add_argument('--phases-dir', default=None,
                    help='rebuilt phases (with index.json) for a run that sent none to a rig -- a sim run; '
                         'written by mppi_viz_data.py --rebuild-phases')
    ap.add_argument('--ghosts', default=None,
                    help='ghost_requests.json from mppi_viz_data.py: solve and export the ghost arms only')
    a = ap.parse_args(argv)
    if a.ghosts:
        os.makedirs(a.out, exist_ok=True)
        ghost_tracks(json.load(open(a.ghosts)), a.config, a.out)
        return 0

    run = os.path.abspath(a.run)
    os.makedirs(a.out, exist_ok=True)
    metrics = json.load(open(os.path.join(run, 'metrics.json')))
    pegs = metrics.get('planning_pegs') or []
    timing = session_timing(run)
    phases = executed_phases(run, metrics, a.phases_dir)
    if a.only:
        phases = [p for p in phases if p['tag'] in a.only]
    print('session timing %s; %d phases' % (timing, len(phases)))

    model = data = qadr = gadr = None
    bases = None
    tracks, sched = {}, {}
    for p in phases:
        phase = json.load(open(p['plan']))
        sol, meta, world = RS.solve_phase(phase, a.config)
        if model is None:
            bases = world
            hose = np.array([[.45, -.3, .785], [.45, .3, .785]])
            model, data, qadr, gadr = SC.build(pegs, hose, world['left'], world['right'])
        elif any(np.abs(world[t] - bases[t]).max() > 1e-9 for t in ARMS):
            raise RuntimeError('%s: base poses differ from the first phase; one world model cannot serve both'
                               % p['tag'])
        worst = RS.verify(model, data, qadr, gadr, sol, log=lambda *_: None)
        moves = schedule(phase, sol, timing, readback=p['grip'])
        t_end = max(m['t1'] for m in moves)
        times = np.arange(0., t_end + 1. / a.rate, 1. / a.rate)
        q = joint_track(moves, times)
        arm_bodies = [i for i in range(model.nbody) if model.body(i).name[:2] in ('L_', 'R_')]
        pos = np.zeros((len(times), len(arm_bodies), 3))
        quat = np.zeros((len(times), len(arm_bodies), 4))
        tool = np.zeros((len(times), 2, 3))
        for k in range(len(times)):
            SC.pose(model, data, qadr, gadr, {TAG[arm]: q[k, i, :6] for i, arm in enumerate(ARMS)},
                    {TAG[arm]: q[k, i, 6] for i, arm in enumerate(ARMS)})
            pos[k] = data.xpos[arm_bodies]
            quat[k] = data.xquat[arm_bodies]
            for i, arm in enumerate(ARMS):
                tool[k, i] = data.site('%s_grasp_site' % TAG[arm]).xpos
        motion = [m for m in moves if m['stage'] == 1]
        mover_arm = meta['rig_arm'][phase.get('mover') or 'minus_y']
        tracks[p['tag']] = dict(t=times, q=q, body_pos=pos, body_quat=quat, tool=tool)
        sched[p['tag']] = dict(
            kind=p['kind'], n=p['n'], attempt=p['attempt'], plan=os.path.basename(p['plan']),
            sync=bool(meta.get('sync', False)), mover_arm=mover_arm, rig_arm=meta['rig_arm'],
            duration_s=float(t_end), chain_worst_mm=float(1e3 * worst),
            motion_window=[float(min(m['t0'] for m in motion)), float(max(m['t1'] for m in motion))]
            if motion else None,
            ik_worst_mm={r: float(s['pos_err_mm'].max()) for r, s in sol.items()},
            grip_source='readback' if p['grip'] else 'commanded',
            moves=[dict(arm=m['arm'], name=m['name'], grip=float(m['grip']), t0=float(m['t0']),
                        t1=float(m['t1']), stage=int(m['stage'])) for m in moves])
        print('%-18s %2d moves over %5.1f s  sync %-5s  whole-chain worst %.2f mm  motion %s'
              % (p['tag'], len(moves), t_end, sched[p['tag']]['sync'], 1e3 * worst,
                 sched[p['tag']]['motion_window']))

    names = [model.body(i).name for i in range(model.nbody) if model.body(i).name[:2] in ('L_', 'R_')]
    graph = build_scene_graph(model, mesh_files=mesh_file_map(model))
    graph['bodies'] = [b for b in graph['bodies'] if b['name'] in names]
    graph['cameras'] = []
    graph['T_world_base'] = {t: np.asarray(bases[t]).tolist() for t in ARMS}
    json.dump(graph, open(os.path.join(a.out, 'robot_scene.json'), 'w'), indent=1)
    arrays = dict(body_names=np.array(names))
    for tag, tr in tracks.items():
        for key, value in tr.items():
            arrays['%s/%s' % (tag, key)] = np.asarray(value, np.float32 if key != 't' else np.float64)
    np.savez_compressed(os.path.join(a.out, 'robot_tracks.npz'), **arrays)
    json.dump(dict(timing=timing, rate_hz=a.rate, home_q='zeros (assumed; not logged)', phases=sched),
              open(os.path.join(a.out, 'robot_schedule.json'), 'w'), indent=1)
    n_geoms = sum(len(b['geoms']) for b in graph['bodies'])
    print('-> %s: %d arm bodies, %d visual geoms, %d phases' % (a.out, len(names), n_geoms, len(tracks)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
