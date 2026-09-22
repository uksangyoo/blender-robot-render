#!/usr/bin/env python3
"""One real routing run -> a frame-exact timeline for the Blender MPPI video. Read-only on the run.

Stage 2 of `render_mppi_run.py` (runs in .venv-v5, after `mppi_viz_robot.py` has written the arm tracks):

    .venv-v5/bin/python scripts/v5/mppi_viz_data.py --run outputs/mppi_real_v5/run_20260917_150620 \
        --bundle outputs/mppi_real_v5/run_20260917_150620/blender

Everything is read through `showcase_data` (observations, samples, `hand_paths`, the run's own evaluator),
so this file adds only the TIMELINE: what is on screen at every output frame, on both panels.

WHAT THE BLENDER PANEL SHOWS, and where each piece comes from
  hose           the tracked 28-node centres the runner planned on (`observations/<tag>.npz` pre/post). Between
                 the two, and only while the arm carries it, the nodes are blended pre -> post, paced by how far
                 along its commanded path the gripper is. No simulated dynamics; the film's own tracker is NOT
                 used mid-move (the arm occludes the hose it carries and the tracker latches onto the arm).
  candidates     every MPPI iteration's rows (`samples/cycle_*_iter_*.npz`): the first planned move of each
                 candidate (the move that would execute now), as the gripper path `hand_paths` decodes. Feasible
                 rows are coloured by their MPPI cost, refused rows are thin grey. A subsample keeps it legible.
  forecasts      the learned model's predicted hose after that first move (`pred[:, 0]`) for the best rows,
                 and `planning.predicted_states[0]` for the plan that ran -- logged, never re-simulated.
  selected plan  the action MPPI chose (`planned_mind_action`: the importance-weighted mean, re-verified).
  rig path       the gripper path the rig actually drove (the phase JSON keyframes, IK -> FK), which adds the
                 carry lift the planner's action does not contain.
  arms           `mppi_viz_robot.py` tracks, placed in film time by a fitted offset (below).
  peg sides      the run's `PegSideEvaluator` on the OBSERVED states only; they change at a result, never mid-move.

HOW MPPI COST MAPS TO COLOUR
  per cycle, over its feasible rows (all iterations): u = (cost - min) / (max - min), colour = cost_colour(u):
  green = lowest cost (good), amber = middle, red = highest (bad); opacity and tube radius also fall with u. `cost` is
  the planner's own objective (`samples['costs']`, what it weighted with) unless --color-by forecast, which
  re-scores each row's forecast with the run's routing objective (`showcase_data.score_predictions`). Equal
  costs get equal colours: tied rows got equal MPPI weight. Magenta is reserved for the plan that ran.

HOW THE TWO PANELS ARE SYNCHRONISED
  Film frames carry the capture clock `t`. In an execution segment every output frame IS one film frame (at
  --probe-speed / --cycle-speed times real time), and the Blender panel is evaluated at that frame's `t`.
  The arm schedule starts at an offset inside the film that the run does not log: it is fitted per phase by
  cross-correlating the film's frame-difference energy with the schedule's joint speed. Planning, intro and
  result segments are pauses of the real clock: the RGB panel holds a frame and says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mppi_paths import HOSE_ROUTING_ROOT

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import showcase_data as D                                                     # noqa: E402
import showcase_data_legacy as DL                                             # noqa: E402

MAGENTA = (1.00, 0.20, 0.62)
REFUSED = (0.60, 0.62, 0.66)
RIG_PATH = (0.96, 0.96, 0.98)
PROBE = (0.45, 0.80, 0.98)
SIDE_WORD = {True: 'behind', False: 'in front of'}


# ----------------------------------------------------------------------------- small helpers

def catmull_rom(points, per_segment=6) -> np.ndarray:
    """A smooth curve through every waypoint (it passes exactly through them)."""
    p = np.asarray(points, np.float64)
    if len(p) < 3:
        return p
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for s in np.linspace(0, 1, per_segment, endpoint=False):
            s2, s3 = s * s, s * s * s
            out.append(.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2
                             + (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
    out.append(p[-1])
    return np.asarray(out)


def smooth01(x):
    x = np.clip(x, 0., 1.)
    return x * x * (3 - 2 * x)


COST_STOPS = ((.16, .78, .40), (.96, .76, .18), (.90, .24, .22))   # u = 0 green, .5 amber, 1 red


def cost_colour(u):
    """Normalised cost -> RGB: green (good) through amber to red (bad). Saturated stops, not RdYlGn, whose
    pale-yellow middle vanishes against the white hose."""
    u = float(np.clip(u, 0., 1.)) * (len(COST_STOPS) - 1)
    i = min(int(u), len(COST_STOPS) - 2)
    a, b = np.asarray(COST_STOPS[i]), np.asarray(COST_STOPS[i + 1])
    return tuple(float(c) for c in a + (u - i) * (b - a))


def film(run: Path, tag: str):
    """A phase's recorded film: frame paths and capture times. -> dict or None"""
    path = run / 'film' / tag / 'frames.jsonl'
    if not path.exists():
        return None
    head, rows = None, []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('kind') == 'phase':
            head = row
        elif 't' in row:
            rows.append(row)
    frames = [run / 'film' / tag / ('f%05d.jpg' % int(r['i'])) for r in rows]
    keep = [i for i, f in enumerate(frames) if f.exists()]
    return dict(t=np.array([rows[i]['t'] for i in keep], np.float64), frames=[frames[i] for i in keep],
                head=head or {})


SIM_FILM_HZ = 30.          # the grid a filmless run's panel is sampled on, in place of a camera's frames


def sim_film(clock: float, duration: float, lead=.5, tail=1.) -> dict:
    """A phase of a run with NO camera, in the shape `film` returns: a clock and no frames.

    A simulated run has nothing to synchronise against, so the arm schedule IS the clock -- offset 0, no fit.
    `clock` is where this phase starts on the video's own running clock, which counts commanded motion only:
    a sim has no wall clock, and the seconds its planner took are shown as pauses, not as elapsed time.
    """
    t = clock + np.arange(0., duration + lead + tail, 1. / SIM_FILM_HZ)
    return dict(t=t, frames=[''] * len(t), head={}, simulated=True)


def motion_energy(frames, size=(240, 135)) -> np.ndarray:
    """Mean absolute grey-level change between consecutive film frames (the first frame reads 0)."""
    import cv2
    out, prev = [], None
    for f in frames:
        g = cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY), size).astype(np.float32)
        out.append(0. if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return np.asarray(out)


def fit_offset(t_film, energy, track_t, track_q, lo=-1.5, hi=5., step=.02):
    """Where the arm schedule starts inside the film. -> (offset s, correlation)

    The film clock and the schedule's clock differ by an unlogged constant (IK solve, session latency).
    The schedule's joint speed (rad/s summed over both arms' 6 joints) is shifted until it best explains
    the film's frame-difference energy. Energy is baseline-subtracted; frame 0 carries no difference.
    """
    speed = np.abs(np.diff(track_q[:, :, :6], axis=0)).sum(axis=(1, 2)) / np.diff(track_t)
    mid = .5 * (track_t[1:] + track_t[:-1])
    rel = t_film - t_film[0]
    e = energy[1:] - np.percentile(energy[1:], 20)
    best = (0., -2.)
    for off in np.arange(lo, hi, step):
        m = np.interp(rel[1:] - off, mid, speed, left=0., right=0.)
        if m.std() < 1e-9:
            continue
        c = float(np.corrcoef(m, e)[0, 1])
        if c > best[1]:
            best = (float(off), c)
    return best


class RigPlanner:
    """A candidate row -> the keyframes the rig would have executed for it, by the runner's own code path.

    `run_real_v5.run_phase` + `YamExecutor.execute`: overshoot, the protocol's carry rule ('lifted' rows are flown
    at carry height), the release clearance, then `act.executor.phase_to_motion_plan` with the run's rig config
    (rig.yaml, hose radius from --cable-radius-m). Checked on this run: the four executed cycles' logged actions
    rebuild their phases/*.json keyframes to 0.00 mm.

    Arms are assigned by geometry -- the grasp with the lower y goes to the -y arm -- never by rebuilding
    PhaseAction(role, a[0], a[1]) from the sorted 20-number vector, which crossed the arms on the rig
    (`run_real_v5`, "Execute THIS, never ..."). The same rule reproduces all four executed assignments.
    """

    def __init__(self, run: Path, data: dict):
        root = HOSE_ROUTING_ROOT
        for path in (root, root / 'mind-cable-v5'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from scripts.mppi import run_real_v5 as R
        from hose_routing.config import RigConfig
        from hose_routing.act.action import PhaseAction
        from hose_routing.act.executor import phase_to_motion_plan
        self.R, self.PhaseAction, self.motion_plan = R, PhaseAction, phase_to_motion_plan
        m = data['metrics']
        self.args = m.get('args') or {}
        self.cfg = RigConfig.load(run / 'config' / 'rig.yaml')
        if self.args.get('cable_radius_m'):
            self.cfg.cable.radius_m = float(self.args['cable_radius_m'])
        self.carry = (m.get('protocol') or {}).get('route_carry') or {}
        self.pegs = m.get('planning_pegs') or []
        first = next((c['plan'] for c in data['cycles'] if c['plan']), None)
        if first is not None:
            self.meta = {k: first['meta'][k] for k in ('T_world_leftbase', 'T_world_rightbase', 'rig_arm',
                                                       'frames_source') if k in first['meta']}
        else:
            # A SIMULATED run: it drove grasp points, never an arm, so no plan file carries the frame graph.
            # Take the one the executor would have shipped -- this config's own, which is the alignment the
            # simulator itself uses (`RigConfig.frames`, `mind-cable/sim/rig_alignment.json`).
            fr = self.cfg.frames()
            self.meta = dict(T_world_leftbase=np.asarray(fr.T_world_leftbase, float).reshape(-1).tolist(),
                             T_world_rightbase=np.asarray(fr.T_world_rightbase, float).reshape(-1).tolist(),
                             rig_arm={'minus_y': fr.rig_arm(0), 'plus_y': fr.rig_arm(1)},
                             frames_source=str(fr.source))

    def phase(self, row, state, role=None):
        row = np.asarray(row, np.float64)          # `role` is deliberately unused: see the class docstring
        w0, w1 = self.R.action_waypoints_m(row)
        n = len(state) - 1
        y0, y1 = state[int(round(row[0] * n)), 1], state[int(round(row[1] * n)), 1]
        return self.PhaseAction(0 if y0 <= y1 else 1, float(row[0]), float(row[1]), np.asarray(w0), np.asarray(w1), 0.)

    def plan(self, phase, state) -> dict:
        R, rc, cfg = self.R, self.carry, self.cfg
        phase = R.overshoot_phase(phase, float(self.args.get('overshoot_xy') or 1.))
        kw = {}
        if rc:
            carried = rc.get('mode', 'all') == 'all' or R.phase_is_lifted(phase, float(rc.get('lift_threshold_m', 0.)))
            release = R.release_clearance(phase, state, self.pegs, float(rc.get('open_clear_m', 0.)),
                                          radius=cfg.cable.radius_m)
            kw = dict(carry=True, carry_wp=int(rc.get('carry_wp', 3)), open_clear=release) if carried else \
                dict(raise_before_open=True, open_clear=release)
        plan = json.loads(self.motion_plan(
            phase, state, cfg.table_z, cfg.cable.radius_m, cfg.approach_lift_m, cfg.release_lift_m, sweep_deg=25.,
            carry_lift=float(self.args.get('carry_lift') or .08), grasp_z_offset=cfg.grasp_z_offset_m, sync=True,
            **kw).to_json())
        plan['meta'].update(self.meta)
        return plan


class LegacyRigPlanner:
    """The same, for the earlier campaign (`run_real_mppi.py`): a candidate row -> its rig keyframes.

    That runner drove every route phase with `PHASE_KW = dict(push=False, sweep=False, carry=False, sync=True)`
    and `single=one_hand(action)` (run_real_mppi.py:88, :171) through the same
    `act.executor.phase_to_motion_plan`, from the repo's `configs/rig.yaml` (the run stored no copy) with the
    open clearance its phases record. Unlike v5 the ROLE is in the record -- `samples['roles'][k, 0]`, 0 = the
    -y arm is the mover (`act/action.py:35-40`) -- so it is read, not inferred from the grasp geometry.

    `verify()` rebuilds every action the rig executed and compares it with the phase JSON the rig was given:
    0.000 mm on all 29 cycles of run_20260911_212456.
    """

    def __init__(self, run: Path, data: dict):
        root = HOSE_ROUTING_ROOT
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from hose_routing.act.action import PhaseAction
        from hose_routing.act.executor import phase_to_motion_plan
        from hose_routing.config import RigConfig
        self.PhaseAction, self.motion_plan = PhaseAction, phase_to_motion_plan
        self.data = data
        self.cfg = RigConfig.load()
        args = data['metrics'].get('args') or {}
        self.sweep_deg = float(args.get('sweep_deg') or 25.)
        self.carry_lift = float(args.get('carry_lift') or .08)
        first = next(c['plan'] for c in data['cycles'] if c['plan'])
        self.open_clear = float(first['meta'].get('open_clear') or 0.)
        self.meta = {k: first['meta'][k] for k in ('T_world_leftbase', 'T_world_rightbase', 'rig_arm',
                                                   'frames_source') if k in first['meta']}

    def phase(self, row, state, role=None):
        row = np.asarray(row, np.float64)
        if role is None:                       # only the executed rows carry one outside the sample files
            n = len(state) - 1
            role = 0 if state[int(round(row[0] * n)), 1] <= state[int(round(row[1] * n)), 1] else 1
        return self.PhaseAction(int(role), float(row[0]), float(row[1]),
                                row[2:11].reshape(3, 3), row[11:20].reshape(3, 3), 0.)

    def plan(self, phase, state) -> dict:
        single = float(phase.sa) == 0. and not np.any(np.asarray(phase.deltas_a, float))  # run_real_mppi.one_hand
        plan = json.loads(self.motion_plan(
            phase, state, self.cfg.table_z, self.cfg.cable.radius_m, self.cfg.approach_lift_m,
            self.cfg.release_lift_m, push=False, sweep=False, carry=False, sync=True, single=single,
            sweep_deg=self.sweep_deg, carry_lift=self.carry_lift, grasp_z_offset=self.cfg.grasp_z_offset_m,
            open_clear=self.open_clear).to_json())
        plan['meta'].update(self.meta)
        return plan

    def verify(self) -> float:
        """Worst keyframe difference, in mm, between a rebuilt executed action and the phase the rig ran."""
        worst = 0.
        for c in self.data['cycles']:
            if c['plan'] is None or c['obs']['executed'] is None:
                continue
            mine = self.plan(self.phase(c['obs']['executed'], c['pre'],
                                        role=int(c['row']['action']['role'])), c['pre'])
            for key in ('minus_y', 'plus_y'):
                a, b = mine.get(key) or [], c['plan'].get(key) or []
                if len(a) != len(b):
                    return float('nan')
                for x, y in zip(a, b):
                    worst = max(worst, float(np.abs(np.asarray(x['position'])
                                                    - np.asarray(y['position'])).max()))
        return 1e3 * worst


class Drawables:
    """Curves the Blender panel animates: tubes with piecewise-linear alpha and grow keys in VIDEO seconds."""

    def __init__(self):
        self.items = []

    def add(self, group, points, rgba, radius, *, alpha=(), grow=(), emission=0., meta=None):
        self.items.append(dict(id=len(self.items), group=group,
                               points=np.round(np.asarray(points, float), 5).tolist(),
                               rgba=[round(float(c), 4) for c in rgba], radius=float(radius),
                               emission=float(emission), alpha=[[round(t, 4), round(a, 4)] for t, a in alpha],
                               grow=[[round(t, 4), round(g, 4)] for t, g in grow], meta=meta or {}))
        return self.items[-1]


# ----------------------------------------------------------------------------- the timeline

class Timeline:
    """Per-output-frame state. Every append is one frame at `fps`."""

    def __init__(self, fps):
        self.fps = fps
        self.hose, self.bpos, self.bquat, self.pegs = [], [], [], []
        self.rgb, self.run_time, self.speed, self.caption = [], [], [], []
        self.captions, self._caption_ids = [], {}
        self.segments = []

    @property
    def now(self):
        return len(self.hose) / self.fps

    def caption_id(self, cap: dict) -> int:
        key = json.dumps(cap, sort_keys=True)
        if key not in self._caption_ids:
            self._caption_ids[key] = len(self.captions)
            self.captions.append(cap)
        return self._caption_ids[key]

    def frame(self, hose, bpos, bquat, pegs, rgb, run_time, speed, caption):
        self.hose.append(np.asarray(hose, np.float32))
        self.bpos.append(np.asarray(bpos, np.float32))
        self.bquat.append(np.asarray(bquat, np.float32))
        self.pegs.append(np.asarray(pegs, np.float32))
        self.rgb.append(str(rgb))
        self.run_time.append(float(run_time))
        self.speed.append(float(speed))
        self.caption.append(self.caption_id(caption))

    def segment(self, kind, label, t0, **extra):
        self.segments.append(dict(kind=kind, label=label, t0=round(t0, 4), t1=round(self.now, 4), **extra))


def overhead_camera(run: Path, frame_path) -> dict:
    """The rig's overhead camera for the whole-chain check: K rescaled to the film's frame size."""
    import cv2
    cam = D.camera(run)
    h, w = cv2.imread(str(frame_path)).shape[:2]
    K = np.asarray(cam['K'], float).copy()
    K[:2] *= w / float(cam['size'][0])
    return dict(K=K.tolist(), T_world_overhead=np.asarray(cam['T_world_overhead']).tolist(), size=[w, h])


def row_role(smp, k):
    """Which arm a sampled row moves, when the record says: `roles` is (K, H) in the earlier campaign and
    (K,) in v5, where the assignment is taken from the grasp geometry instead (`RigPlanner.phase`)."""
    roles = np.asarray(smp['roles'])
    return int(roles[k, 0]) if roles.ndim > 1 else None


def peg_status(ev, state):
    return np.asarray(ev.evaluate(np.asarray(state)[None])['sides'][0], np.float32)


def sides_text(status):
    return ' '.join('p%d %s' % (i, '✓' if s > .5 else '✗') for i, s in enumerate(status))


def hose_during(pre, post, t_rel, window, tool_t, tool_path):
    """Observed pre -> observed post, blended only while the carry runs, paced by the gripper's path length."""
    if window is None or post is None:
        return pre
    lo, hi = window
    if t_rel <= lo:
        return pre
    if t_rel >= hi:
        return post
    sel = (tool_t >= lo) & (tool_t <= hi)
    seg = np.linalg.norm(np.diff(tool_path[sel], axis=0), axis=-1).sum(axis=-1)
    cum = np.concatenate([[0.], np.cumsum(seg)])
    u = np.interp(t_rel, tool_t[sel], cum / cum[-1]) if cum[-1] > 1e-6 else (t_rel - lo) / (hi - lo)
    return pre + smooth01(u) * (post - pre)


def build(run: Path, bundle: Path, a) -> dict:
    data = D.load(run)
    ev = D.check(data)                               # raises unless the evaluator reproduces the record
    metrics = data['metrics']
    events = [json.loads(l) for l in (run / 'events.jsonl').read_text().splitlines() if l.strip()] \
        if (run / 'events.jsonl').exists() else []
    t_start = next((e['t'] for e in events if e['kind'] == 'start'), None)
    sched = json.loads((bundle / 'robot_schedule.json').read_text())['phases']
    tracks = np.load(bundle / 'robot_tracks.npz')
    body_names = [str(n) for n in tracks['body_names']]
    fps = a.fps
    n_pegs = len(data['pegs'])

    # --- phases in execution order, with film, observation, robot track and fitted film offset
    phases = []
    obs_by_tag = {'probe_%03d' % r['n']: r['obs'] for r in data['probes']}
    obs_by_tag.update({'cycle_%03d' % c['n']: c['obs'] for c in data['cycles']})
    filmed = (run / 'film').is_dir() and any((run / 'film').glob('*/frames.jsonl'))
    clock = 0.
    for tag, s in sched.items():
        f = film(run, tag)
        obs = obs_by_tag.get(tag) if data.get('legacy') else D.observation(run, tag)
        if obs is None or obs['pre'] is None or (f is None and filmed):
            print('skip %s (film or observation missing)' % tag)
            continue
        tr = {k: tracks['%s/%s' % (tag, k)] for k in ('t', 'q', 'body_pos', 'body_quat', 'tool')}
        if f is None:                        # a simulated run: no camera, so nothing to fit an offset against
            f = sim_film(clock, float(s['duration_s']))
            clock = float(f['t'][-1]) + 1.
            off, corr = 0., float('nan')
            print('%-18s no film: the schedule is the clock (%5.1f s)' % (tag, s['duration_s']))
        else:
            energy = motion_energy(f['frames'])
            off, corr = fit_offset(f['t'], energy, tr['t'], tr['q'])
            print('%-18s film %5.1f s (%3d frames)  schedule %5.1f s  offset %+5.2f s  r %.2f'
                  % (tag, f['t'][-1] - f['t'][0], len(f['t']), s['duration_s'], off, corr))
        phases.append(dict(tag=tag, sched=s, film=f, obs=obs, track=tr, offset=off, corr=corr))
    if t_start is None:                  # the earlier campaign logged no events: the film is the only clock
        t_start = float(phases[0]['film']['t'][0])
    home_pos = phases[0]['track']['body_pos'][0]
    home_quat = phases[0]['track']['body_quat'][0]

    tl = Timeline(fps)
    dr = Drawables()
    ghosts, requests = [], []
    rigplan = None if a.strokes else (LegacyRigPlanner(run, data) if data.get('legacy')
                                      else RigPlanner(run, data))
    rebuild_mm = rigplan.verify() if isinstance(rigplan, LegacyRigPlanner) else None
    if rebuild_mm is not None:
        print('rebuilt executed actions match the phases the rig ran to %.3f mm' % rebuild_mm)

    def ghost(group, plan, rgba, emission, alpha, motion, **meta):
        gid = len(ghosts)
        ghosts.append(dict(id=gid, group=group, rgba=[round(float(c), 4) for c in rgba], emission=float(emission),
                           alpha=[[round(t, 4), round(v, 4)] for t, v in alpha],
                           motion=[[round(t, 4), round(v, 4)] for t, v in motion], meta=meta))
        requests.append(dict(id=gid, phase=plan))
        return gid
    shown = {'hose': np.asarray(phases[0]['obs']['pre'])}

    def hold(seconds, hose, pegs, rgb, run_time, caption):
        for _ in range(max(1, int(round(seconds * fps)))):
            tl.frame(hose, home_pos, home_quat, pegs, rgb, run_time, 0., caption)
        shown['hose'] = np.asarray(hose)

    def execute(p, speed, caption_fn, pegs_before):
        """Play one phase's film; the Blender panel follows the same capture clock."""
        f, tr, s = p['film'], p['track'], p['sched']
        pre, post = np.asarray(p['obs']['pre']), np.asarray(p['obs']['post'])
        t0f, t1f = f['t'][0], f['t'][-1]
        n = int(np.ceil((t1f - t0f) / speed * fps)) + 1
        blend0 = shown['hose']
        for k in range(n):
            tf = min(t0f + k * speed / fps, t1f)
            i = int(np.searchsorted(f['t'], tf, side='right') - 1)
            t_rel = tf - t0f - p['offset']
            j = int(np.clip(np.searchsorted(tr['t'], t_rel), 0, len(tr['t']) - 1))
            hose = hose_during(pre, post, t_rel, s['motion_window'], tr['t'], tr['tool'])
            if k < int(.4 * fps):
                hose = blend0 + smooth01(k / (.4 * fps)) * (hose - blend0)
            tl.frame(hose, tr['body_pos'][j], tr['body_quat'][j], pegs_before, f['frames'][i],
                     f['t'][i] - t_start, speed, caption_fn(t_rel))
        shown['hose'] = post
        return post

    def film_time_to_video(p, t_rel, t_video0, speed):
        """Schedule-relative time inside a phase -> video seconds, for keys on drawables."""
        return t_video0 + (t_rel + p['offset']) / speed

    def rig_path_drawable(p, t_video0, speed, alpha_end):
        s, tr = p['sched'], p['track']
        if s['motion_window'] is None:
            return
        lo, hi = s['motion_window']
        for arm, idx in (('left', 0), ('right', 1)):
            sel = (tr['t'] >= lo - 1e-6) & (tr['t'] <= hi + 1e-6)
            path = tr['tool'][sel, idx]
            if np.linalg.norm(path[-1] - path[0]) < .01 and np.ptp(path, axis=0).max() < .01:
                continue
            path = path[::3]
            v0 = film_time_to_video(p, lo, t_video0, speed)
            v1 = film_time_to_video(p, hi, t_video0, speed)
            dr.add('rig_path', path, RIG_PATH + (.95,), .0040, emission=.8,
                   grow=[(v0, 0.), (v1, 1.)], alpha=[(v0 - .01, 0.), (v0, 1.), (alpha_end, 1.), (alpha_end + .5, 0.)],
                   meta=dict(tag=p['tag'], arm=arm))

    # ------------------------------------------------------------- intro
    probes = [p for p in phases if p['sched']['kind'] == 'probe']
    cycles = [p for p in phases if p['sched']['kind'] == 'cycle']
    first = phases[0]
    status0 = peg_status(ev, first['obs']['pre'])
    if data.get('legacy'):
        landmarks = DL.landmarks(data, ev)
    else:
        from hose_routing.v5 import goal_suite as GS
        contract, _target = GS.goal_contract(data['task_live'])
        landmarks = [dict(peg=int(l['peg_id']), point=[float(v) for v in l['point']],
                          normal=[float(v) for v in l['normal']], behind=bool(l['normal'][0] > 0))
                     for l in contract.landmarks]
    goal_words = ', '.join('%s p%d' % (SIDE_WORD[l['behind']], l['peg']) for l in landmarks)
    cost_label = str(data.get('cost_label') or 'routing cost')
    cost_unit = str(data.get('cost_unit') or '')
    fmt_cost = (lambda v: '%.0f mm' % v) if cost_unit == 'mm' else (lambda v: '%.2f' % v)
    task_name = Path(str((metrics.get('args') or {}).get('task_file') or data['task'].get('task_id', ''))).stem
    task_line = '%s  ·  %d mm hose  ·  %d pegs, %d mm tall' % (
        task_name, round(2000 * data['radius']),
        n_pegs, round(1000 * data['pegs'][0]['h']))
    lm_pegs = [int(l['peg']) for l in landmarks]

    def peg_vector(status):
        """Landmark statuses -> one value per peg, which is how the render indexes them (`A['pegs'][f, peg]`).

        A peg the contract says nothing about (this campaign routes past three of its four) reads 1: no
        marker is drawn for it, and the caption strip lists the landmarks, not this vector.
        """
        out = np.ones(n_pegs, np.float32)
        for i, peg in enumerate(lm_pegs):
            out[peg] = status[i]
        return out

    t0 = tl.now
    hold(a.intro, first['obs']['pre'], peg_vector(status0), first['film']['frames'][0], first['film']['t'][0] - t_start,
         dict(mode='intro', title='Real rig, MPPI with a learned hose model', stage=task_line,
              detail='goal: route the hose %s (as seen from the robot)' % goal_words, sides=status0.tolist()))
    tl.segment('intro', 'intro', t0)

    # ------------------------------------------------------------- calibration probes
    n_probe = max([p['sched']['n'] for p in probes] or [0])
    for p in probes:
        s = p['sched']
        pre = p['obs']['pre']
        st_pre = peg_status(ev, pre)
        row = next((r for r in metrics['probes'] if int(r['k']) == s['n'] and int(r.get('attempt', 0) or 0)
                    == s['attempt']), {})
        slipped = bool(row.get('slipped'))
        label = 'calibration probe %d / %d' % (s['n'], n_probe) + (' (retry)' if s['attempt'] else '')
        detail = 'a small known move; its effect fits the model\'s latent z'
        if slipped:
            detail = 'the hose slipped in the jaws (%s): not used, retried' % '; '.join(row.get('reasons') or [])
        t0 = tl.now
        # the commanded probe move, as the planner's action decodes it, over the observed hose
        nodes, paths = D.hand_paths(p['obs']['executed'] if p['obs']['executed'] is not None
                                    else p['obs']['planned'], pre)
        lo, hi = s['motion_window'] or (0., s['duration_s'])
        v0 = film_time_to_video(p, lo - 1.2, t0, a.probe_speed)
        v_end = t0 + (p['film']['t'][-1] - p['film']['t'][0]) / a.probe_speed
        for h in np.flatnonzero(D.moved(paths)) if a.strokes else ():
            dr.add('probe', catmull_rom(paths[h]), PROBE + (.9,), .0055, emission=.8,
                   grow=[(v0, 0.), (v0 + .5, 1.)], alpha=[(v0 - .01, 0.), (v0, 1.), (v_end, 1.), (v_end + .4, 0.)],
                   meta=dict(tag=p['tag'], hand=int(h)))
        if a.strokes:
            rig_path_drawable(p, t0, a.probe_speed, v_end)
        execute(p, a.probe_speed, lambda _t, label=label, detail=detail: dict(
            mode='probe', title=label, stage='executing on the rig', detail=detail, sides=st_pre.tolist(),
            legend='probe' if a.strokes else None),
            peg_vector(st_pre))
        tl.segment('probe', label, t0, tag=p['tag'], speed=a.probe_speed, offset_s=p['offset'],
                   offset_r=p['corr'], slipped=slipped)

    zfit = next((e for e in events if e['kind'] == 'z_fit'), None)
    if zfit and probes and cycles:
        rep = zfit.get('report') or {}
        fitted, nomo = np.mean(rep.get('fitted_mean_mm') or [np.nan]), np.mean(rep.get('nomotion_mean_mm') or [np.nan])
        last = probes[-1]
        t0 = tl.now
        hold(a.zfit, last['obs']['post'], peg_vector(peg_status(ev, last['obs']['post'])), last['film']['frames'][-1],
             zfit['t'] - t_start, dict(mode='zfit', title='model calibrated on %d probes' % zfit.get('rows', len(probes)),
                                       stage='latent z fitted to what the probes did',
                                       detail='forecast error %.0f mm vs %.0f mm for "the hose does not move"' % (fitted, nomo),
                                       sides=peg_status(ev, last['obs']['post']).tolist()))
        tl.segment('zfit', 'z fit', t0)

    # ------------------------------------------------------------- MPPI cycles
    rng = np.random.default_rng(a.seed)
    cyc_rows = {c['n']: c for c in data['cycles']}
    cycle_report = []
    for p in cycles:
        n = p['sched']['n']
        c = cyc_rows[n]
        row, smp = c['row'], c['samples']
        pre, post = np.asarray(c['pre']), np.asarray(c['post'])
        if smp is not None and np.abs(smp['state'] - pre).max() > 1e-3:
            print('WARNING cycle %d: samples state differs from observation pre by %.1f mm'
                  % (n, 1e3 * np.abs(smp['state'] - pre).max()))
        st_pre, st_post = peg_status(ev, pre), peg_status(ev, post)
        plan_ev = next((e for e in events if e['kind'] == 'plan' and e.get('cycle') == n), {})
        plan = row.get('planning') or {}
        # how long this cycle's planning took: the run's own events, or the cycle row when there are none
        plan_ev = dict(plan_ev)
        plan_ev.setdefault('planning_seconds', plan.get('planning_seconds') or row.get('planning_seconds'))
        n_iter = smp['iterations'] if smp is not None else 0
        title = 'MPPI cycle %d / %d' % (n, len(cycles))
        rgb_plan = p['film']['frames'][0]
        clock_plan = (plan_ev.get('t', p['film']['t'][0]) - t_start)
        cost_word = 'MPPI cost' if a.color_by == 'planner' else 'forecast routing cost'

        # colour value per row: the planner's own objective (or the re-scored forecast)
        if smp is not None:
            valid = smp['valid'].copy()
            cost = smp['costs'].astype(float)
            if a.color_by == 'forecast':
                cost = D.score_predictions(ev, smp['pred'][:, 0 if a.forecast_step == 'first' else -1])['cost']
            has_pred = np.isfinite(smp['pred'][:, 0]).all(axis=(1, 2)) if smp['pred'].shape[-1] == 3 \
                else np.zeros(len(cost), bool)
            drawable = valid & has_pred          # v5 keeps every row's forecast, so this is `valid` there
            lo_c, hi_c = (float(cost[valid].min()), float(cost[valid].max())) if valid.any() else (0., 1.)
            u = np.clip((cost - lo_c) / max(hi_c - lo_c, 1e-9), 0, 1) if hi_c > lo_c else np.zeros(len(cost))
        t_plan0 = tl.now
        t_iter0 = t_plan0 + a.plan_intro
        t_fore = t_iter0 + n_iter * a.plan_iter
        t_sel = t_fore + a.plan_forecast
        t_exec = t_sel + a.plan_select
        exec_dur = (p['film']['t'][-1] - p['film']['t'][0]) / a.cycle_speed
        t_result = t_exec + exec_dur
        t_end = t_result + a.result

        drawn_valid, focus_pts, ghost_rows = [], [], []
        for it in range(n_iter):
            m = smp['iteration'] == it
            ti, tn = t_iter0 + it * a.plan_iter, t_iter0 + (it + 1) * a.plan_iter
            last_iter = it == n_iter - 1
            ok = np.flatnonzero(m & drawable)
            ok = ok[np.argsort(cost[ok], kind='stable')]
            hold_row = np.flatnonzero(m)[-1]
            ok = ok[ok != hold_row]                       # the explicit hold-still row has no path to draw
            top = ok[:a.top_n]
            rest = ok[a.top_n:]
            if len(rest) > a.spread_n:
                rest = rest[np.unique(np.linspace(0, len(rest) - 1, a.spread_n).round().astype(int))]
            chosen = np.concatenate([top, rest])
            if not a.strokes:
                # GHOST ARMS: a spread of the ranking (best included), each flown as the rig would have flown it
                pick = ok[np.unique(np.linspace(0, len(ok) - 1, min(a.ghost_n, len(ok))).round().astype(int))] \
                    if len(ok) else ok
                order = pick[np.argsort(-u[pick], kind='stable')]              # worst first, best last
                for r, k in enumerate(order):
                    plan = rigplan.plan(rigplan.phase(smp['actions'][k, 0], smp['state'],
                                                      role=row_role(smp, k)), smp['state'])
                    start = ti + .1 + 1.0 * r / max(1, len(order) - 1)
                    keys = [(start - .01, 0.), (start + .25, 1.)]
                    # the last iteration's arms stay, dimmed, while their forecast hoses come up in front of them
                    keys += [(t_fore, 1.), (t_fore + .4, .35), (t_sel, .35), (t_sel + .5, 0.)] if last_iter else \
                        [(tn - .05, 1.), (tn + .35, 0.)]
                    ghost('candidate', plan, cost_colour(u[k]) + (.40 - .12 * u[k],), .9, keys,
                          [(start + .25, 0.), (start + 1.25, 1.)], cycle=n, iteration=it, row=int(k),
                          cost=float(cost[k]), u=float(u[k]), family=str(smp['family'][k]))
                    for key in ('minus_y', 'plus_y'):
                        focus_pts.extend(np.asarray([kf['position'] for kf in plan[key]
                                                     if kf['name'] not in ('approach', 'retreat')]).reshape(-1, 3))
                    if last_iter:
                        ghost_rows.append(k)
                drawn_valid.extend(int(k) for k in order)
                continue
            bad = np.flatnonzero(m & ~valid)
            bad = rng.choice(bad, size=min(a.refused_n, len(bad)), replace=False) if len(bad) else bad
            # refused rows: thin grey, gone when the next iteration starts
            for k in bad:
                nodes, paths = D.hand_paths(smp['actions'][k, 0], smp['state'])
                start = ti + rng.uniform(0, .8)
                for h in np.flatnonzero(D.moved(paths)):
                    dr.add('refused', catmull_rom(paths[h]), REFUSED + (.40,), .0018,
                           grow=[(start, 0.), (start + .45, 1.)],
                           alpha=[(start - .01, 0.), (start, 1.), (tn - .1, 1.), (tn + .3, 0.)],
                           meta=dict(cycle=n, iteration=it, row=int(k), refused=str(smp['refused'][k])))
            # feasible rows: worst appear first, best last; dim when the next iteration arrives
            order = chosen[np.argsort(-u[chosen], kind='stable')]
            for r, k in enumerate(order):
                nodes, paths = D.hand_paths(smp['actions'][k, 0], smp['state'])
                start = ti + .15 + 1.25 * r / max(1, len(order) - 1)
                col = cost_colour(u[k])
                alpha_hi = .95 - .5 * u[k]
                keys = [(start - .01, 0.), (start, 1.)]
                # near-identical rows stack: 30 strokes at 8% read as opaque, so dimmed layers stay faint and
                # everything but the chosen plan is gone once it is selected
                if not last_iter:
                    keys += [(tn - .05, 1.), (tn + .35, .10), (t_fore, .10), (t_fore + .4, 0.)]
                else:
                    keys += [(t_fore, 1.), (t_fore + .4, .35), (t_sel, .35), (t_sel + .6, 0.)]
                for h in np.flatnonzero(D.moved(paths)):
                    dr.add('candidate', catmull_rom(paths[h]), col + (alpha_hi,), .0030 + .0020 * (1 - u[k]),
                           emission=.15 + .35 * (1 - u[k]), grow=[(start, 0.), (start + .55, 1.)], alpha=keys,
                           meta=dict(cycle=n, iteration=it, row=int(k), cost=float(cost[k]), u=float(u[k]),
                                     family=str(smp['family'][k])))
                    focus_pts.append(paths[h])
                drawn_valid.append(k)

        # the model's forecasts for the best rows (last iteration's ranking pooled over all iterations)
        if smp is not None and drawable.any():
            ok = np.flatnonzero(drawable)
            ok = ok[np.argsort(cost[ok], kind='stable')]
            picks = list(ok[:a.forecast_top])
            others = ok[a.forecast_top:]
            if len(others):
                picks += list(others[np.unique(np.linspace(0, len(others) - 1, a.forecast_spread).round().astype(int))])
            if not a.strokes:                    # each held ghost arm gets the hose the model forecast for it
                picks = sorted(ghost_rows, key=lambda k: cost[k])
            for r, k in enumerate(picks[::-1]):
                start = t_fore + .9 * r / max(1, len(picks) - 1)
                dr.add('forecast', smp['pred'][k, 0],
                       cost_colour(u[k]) + (.7,), .0055, emission=.25,
                       grow=[(start, 0.), (start + .5, 1.)],
                       alpha=[(start - .01, 0.), (start, 1.), (t_sel, 1.), (t_sel + .5, 0.)],
                       meta=dict(cycle=n, row=int(k), cost=float(cost[k])))

        # the plan that ran: its first move and the model's forecast for it
        planned = c['obs']['planned']
        nodes, paths = D.hand_paths(planned, pre)
        if not a.strokes and c['plan']:
            ghost('selected', c['plan'], MAGENTA + (.62,), .9,
                  [(t_sel - .01, 0.), (t_sel + .3, 1.), (t_exec, 1.), (t_exec + .7, 0.)],
                  [(t_sel + .3, 0.), (t_sel + 1.6, 1.)], cycle=n, phase=Path(str(row.get('plan_file'))).name)
        for h in np.flatnonzero(D.moved(paths)) if a.strokes else ():
            dr.add('selected', catmull_rom(paths[h]), MAGENTA + (1.,), .0085, emission=4.,
                   grow=[(t_sel, 0.), (t_sel + .8, 1.)],
                   alpha=[(t_sel - .01, 0.), (t_sel, 1.), (t_result + a.result - .5, 1.), (t_end, 0.)],
                   meta=dict(cycle=n, hand=int(h), node=int(nodes[h])))
        if len(c['predicted']):
            dr.add('selected_forecast', c['predicted'][0], MAGENTA + (.22,), .8 * data['radius'], emission=.4,
                   alpha=[(t_sel + .2, 0.), (t_sel + 1., 1.), (t_result + a.result - .5, 1.), (t_end, 0.)],
                   meta=dict(cycle=n))
        if a.strokes:
            rig_path_drawable(p, t_exec, a.cycle_speed, t_result + a.result - .5)
        subject = 'sample' if a.strokes else 'ghost arm'
        legend_sel = ([[list(MAGENTA), 'magenta = the plan MPPI chose'],
                       [[1., .55, .78], "magenta ghost = the model's forecast of the hose"],
                       [list(RIG_PATH), "white = the path the rig's gripper drove"]] if a.strokes else
                      [[list(MAGENTA), 'magenta arm = the plan MPPI chose'],
                       [[1., .55, .78], "magenta hose = the model's forecast"]])
        legend_mppi = [[list(REFUSED), 'grey = refused samples (infeasible)']] if a.strokes else []

        # frames: planning (real clock paused), execution (film), result (hold)
        feasible = [int((smp['valid'] & (smp['iteration'] == it)).sum()) for it in range(n_iter)] if smp else []
        per_iter = [int((smp['iteration'] == it).sum()) for it in range(n_iter)] if smp else []
        best_iter = [float(np.min(cost[(smp['iteration'] == it) & smp['valid']]))
                     if ((smp['iteration'] == it) & smp['valid']).any() else float('nan') for it in range(n_iter)]
        choice = 'importance-weighted mean of the lowest-cost rows' if plan.get('choice') == 'mean' else \
            str(plan.get('choice_detail') or plan.get('choice'))
        caps = []
        caps.append((t_iter0, dict(mode='plan', title=title, stage='planning from the observed hose',
                                   detail='%s %s  ·  %s' % (cost_label, fmt_cost(row['cost_before']),
                                                            sides_text(st_pre)),
                                   sides=st_pre.tolist(), paused=True, planning_s=plan_ev.get('planning_seconds'))))
        for it in range(n_iter):
            caps.append((t_iter0 + (it + 1) * a.plan_iter, dict(
                mode='plan', title=title, stage='sampling: iteration %d of %d' % (it + 1, n_iter),
                detail='%d candidates  ·  %d feasible  ·  lowest %s %.2f' % (per_iter[it], feasible[it], cost_word,
                                                                            best_iter[it]),
                sides=st_pre.tolist(), legend='mppi', paused=True, planning_s=plan_ev.get('planning_seconds'),
                cost_range=[lo_c, hi_c], cost_word=cost_word, subject=subject, legend_lines=legend_mppi)))
        caps.append((t_sel, dict(mode='plan', title=title, stage='forecasts of the best candidates',
                                 detail='the learned model\'s predicted hose after each move',
                                 sides=st_pre.tolist(), legend='mppi', paused=True,
                                 planning_s=plan_ev.get('planning_seconds'), cost_range=[lo_c, hi_c],
                                 cost_word=cost_word, subject=subject, legend_lines=legend_mppi)))
        caps.append((t_exec, dict(mode='plan', title=title, stage='selected: %s' % choice,
                                  detail='family %s  ·  forecast %s %s' % (c['family'], cost_label,
                                                                          fmt_cost(row['predicted_cost'])),
                                  sides=st_pre.tolist(), legend='selected', paused=True,
                                  planning_s=plan_ev.get('planning_seconds'), legend_lines=legend_sel)))
        while tl.now < t_exec - 1e-9:
            cap = next(cp for tt, cp in caps if tl.now < tt - 1e-9)
            tl.frame(pre if tl.now > t_plan0 + .4 else shown['hose'] + smooth01((tl.now - t_plan0) / .4) * (pre - shown['hose']),
                     home_pos, home_quat, peg_vector(st_pre), rgb_plan, clock_plan, 0., cap)
        shown['hose'] = pre
        focus = np.vstack(focus_pts).mean(axis=0) if focus_pts else pre.mean(axis=0)
        tl.segment('plan', title + ' planning', t_plan0, cycle=n, iterations=n_iter,
                   focus=[float(v) for v in focus])
        t0 = tl.now
        execute(p, a.cycle_speed, lambda t_rel: dict(
            mode='exec', title=title, stage='executing on the rig',
            detail='magenta: the plan  ·  ghost: the forecast', sides=st_pre.tolist(), legend='selected',
            legend_lines=legend_sel[1:]), peg_vector(st_pre))
        tl.segment('exec', title + ' execution', t0, cycle=n, tag=p['tag'], speed=a.cycle_speed,
                   offset_s=p['offset'], offset_r=p['corr'])
        t0 = tl.now
        res = dict(mode='result', title=title, stage='result: %s %s → %s  (forecast %s)'
                   % (cost_label, fmt_cost(row['cost_before']), fmt_cost(row['final_cost']),
                      fmt_cost(row['predicted_cost'])),
                   detail='peg sides  %s' % sides_text(st_post), sides=st_post.tolist(), legend='result',
                   legend_lines=legend_sel[1:])
        n_res = int(round(a.result * fps))
        pv_pre, pv_post = peg_vector(st_pre), peg_vector(st_post)
        for k in range(n_res):
            mix = smooth01(k / max(1, int(.6 * fps)))
            tl.frame(post, home_pos, home_quat, pv_pre + mix * (pv_post - pv_pre), p['film']['frames'][-1],
                     p['film']['t'][-1] - t_start, 0., res)
        tl.segment('result', title + ' result', t0, cycle=n)
        cycle_report.append(dict(cycle=n, cost_before=row['cost_before'], cost_after=row['final_cost'],
                                 forecast=row['predicted_cost'], color_range=[lo_c, hi_c],
                                 drawn_valid=len(drawn_valid)))

    # ------------------------------------------------------------- outro
    final = np.asarray(cycles[-1]['obs']['post']) if cycles else shown['hose']
    st_final = peg_status(ev, final)
    rf = metrics.get('routing_final') or {}
    t0 = tl.now
    hold(a.outro, final, peg_vector(st_final),
         cycles[-1]['film']['frames'][-1] if cycles else first['film']['frames'][-1],
         cycles[-1]['film']['t'][-1] - t_start if cycles else 0.,
         dict(mode='outro', title='routed' if data['success'] else 'not routed',
              stage='all %d peg sides correct  ·  %s %s → %s in %d cycles' % (
                  len(landmarks), cost_label, fmt_cost(data['initial_cost']), fmt_cost(data['final_cost']),
                  len(cycles)) if data['success'] else
              '%s %s → %s' % (cost_label, fmt_cost(data['initial_cost']), fmt_cost(data['final_cost'])),
              detail='held on %d settled observations' % len(cycles[-1]['obs']['settled'])
              if cycles and cycles[-1]['obs']['settled'] is not None else '', sides=st_final.tolist()))
    tl.segment('outro', 'outro', t0)

    # ------------------------------------------------------------- camera: a slow orbit around the board
    nf = len(tl.hose)
    if data.get('legacy'):                       # a captured target shape, held on screen for the whole video
        dr.add('goal', np.asarray(data['target'], float), (.30, .72, 1., .30), .0035, emission=.35,
               meta=dict(what='the captured goal shape this run routes to'))
    tt = np.arange(nf) / fps
    # while a cycle plans, ease in toward where its candidates are; ease back out as the rig starts moving
    w, focus = np.zeros(nf), np.repeat(np.array(a.cam_target, float)[None], nf, 0)
    for sgm in tl.segments:
        if sgm['kind'] != 'plan':
            continue
        ramp = smooth01((tt - sgm['t0']) / 1.4) * (1 - smooth01((tt - sgm['t1'] - .3) / 1.6))
        on = ramp > w
        w[on] = ramp[on]
        f = np.array(sgm['focus'], float)
        f[2] = a.cam_target[2]
        focus[on] = f
    # whole ghost arms need the whole board in view: only a gentle ease-in when they are drawn
    zoom, lean = (a.cam_zoom, a.cam_focus) if a.strokes else (a.ghost_cam_zoom, a.ghost_cam_focus)
    target = np.array(a.cam_target, float)[None] + (lean * w)[:, None] * (focus - np.array(a.cam_target))
    dist = a.cam_distance * (1 - w * (1 - zoom))
    az = np.radians(a.cam_azimuth + a.cam_orbit * np.sin(2 * np.pi * tt / a.cam_period))
    el = np.radians(a.cam_elevation)
    eye = target + dist[:, None] * np.stack([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az),
                                              np.full(nf, np.sin(el))], axis=1)
    out = dict(
        run=str(run), fps=fps, n_frames=nf, duration_s=nf / fps, color_by=a.color_by,
        legacy=bool(data.get('legacy')), cost_label=cost_label, cost_unit=cost_unit,
        rebuild_mm=rebuild_mm, cost_axis_checked=int(data.get('cost_axis_checked', 0)),
        table_z=data['table_z'], hose_radius=data['radius'], pegs=data['pegs'], landmarks=landmarks,
        lens_mm=a.cam_lens, rgb_size=list(D.camera(run)['size']) if filmed else [1280, 720],
        filmed=bool(filmed), body_names=body_names,
        captions=tl.captions, segments=tl.segments, rgb=tl.rgb, drawables=dr.items, ghosts=ghosts,
        sync=[dict(tag=p['tag'], offset_s=p['offset'], r=p['corr'],
                   film_s=float(p['film']['t'][-1] - p['film']['t'][0]),
                   schedule_s=p['sched']['duration_s']) for p in phases] if filmed else [],
        cycles=cycle_report, colors=dict(selected=MAGENTA, refused=REFUSED, rig_path=RIG_PATH, probe=PROBE,
                                         colormap='green (low cost) - amber - red (high cost)'),
        colorbar=[cost_colour(i / 63) for i in range(64)],
        overhead=overhead_camera(run, phases[0]['film']['frames'][0]) if filmed else None)
    (bundle / 'timeline.json').write_text(json.dumps(out))
    (bundle / 'ghost_requests.json').write_text(json.dumps(requests))
    np.savez_compressed(bundle / 'timeline.npz', hose=np.stack(tl.hose), body_pos=np.stack(tl.bpos),
                        body_quat=np.stack(tl.bquat), pegs=np.stack(tl.pegs), run_time=np.array(tl.run_time),
                        speed=np.array(tl.speed), caption=np.array(tl.caption, np.int32), cam_eye=eye.astype(np.float32),
                        cam_target=target.astype(np.float32))
    groups = {}
    for d in dr.items:
        groups[d['group']] = groups.get(d['group'], 0) + 1
    print('timeline: %d frames = %.1f s at %d fps; drawables %s; ghost arms %d' % (nf, nf / fps, fps, groups,
                                                                                    len(ghosts)))
    for sgm in tl.segments:
        print('  %6.1f-%6.1f  %-8s %s' % (sgm['t0'], sgm['t1'], sgm['kind'], sgm['label']))
    return out


def rebuild_phases(run: Path, out: Path) -> dict:
    """Write the phase each executed action WOULD have sent to the rig, for a run that sent none.

    A `--rig sim` run moves grasp points in Newton; nothing solves an arm, so `phases/` is empty and the video
    would have no arms at all. The keyframes here come from the same path every other plan in this pipeline
    takes -- the runner's overshoot and carry rule, then `act.executor.phase_to_motion_plan` with the run's own
    rig.yaml and hose radius -- so they are the commanded gripper waypoints the sim executed, with an arm
    behind them that the rig's IK puts there. That is a reconstruction, and the video's README says so.
    """
    data = D.load(run)
    D.check(data)
    planner = RigPlanner(run, data)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    rows = [('probe_%03d' % int(r['row'].get('k', r['n'])) + ('_retry%d' % int(r['row'].get('attempt', 0) or 0)
                                                              if r['row'].get('attempt') else ''), r)
            for r in data['probes']] + [('cycle_%03d' % c['n'], c) for c in data['cycles']]
    for tag, row in rows:
        act = row['obs'].get('executed') if row.get('obs') else None
        if act is None or not bool(row['row'].get('executed', True)):
            continue
        plan = planner.plan(planner.phase(np.asarray(act, float), np.asarray(row['pre'], float)),
                            np.asarray(row['pre'], float))
        path = out / ('%s.json' % tag)
        path.write_text(json.dumps(plan, indent=1))
        index[tag] = str(path)
    (out / 'index.json').write_text(json.dumps(index, indent=1))
    print('rebuilt %d executed phases into %s (this run sent none to a rig)' % (len(index), out))
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--bundle', default=None, help='directory with robot_*.json/npz (default <run>/blender)')
    ap.add_argument('--rebuild-phases', default=None, metavar='DIR',
                    help='only rebuild the executed phases of a run that sent none to a rig (a sim run), '
                         'into DIR, and stop; the robot stage then solves them with --phases-dir')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--color-by', choices=('planner', 'forecast'), default='planner')
    ap.add_argument('--forecast-step', choices=('first', 'last'), default='last',
                    help='with --color-by forecast: score the forecast after the first move or the whole horizon')
    ap.add_argument('--strokes', action='store_true',
                    help='draw candidates as gripper-path lines (the first version) instead of ghost arms')
    ap.add_argument('--ghost-n', type=int, default=6, help='ghost arms per MPPI iteration, spread over the ranking')
    ap.add_argument('--top-n', type=int, default=24, help='lowest-cost feasible rows drawn per iteration')
    ap.add_argument('--spread-n', type=int, default=24, help='further feasible rows, spread over the cost range')
    ap.add_argument('--refused-n', type=int, default=40, help='refused rows drawn per iteration (grey)')
    ap.add_argument('--forecast-top', type=int, default=8)
    ap.add_argument('--forecast-spread', type=int, default=4)
    ap.add_argument('--probe-speed', type=float, default=5.)
    ap.add_argument('--cycle-speed', type=float, default=2.)
    ap.add_argument('--intro', type=float, default=3.5)
    ap.add_argument('--zfit', type=float, default=2.5)
    ap.add_argument('--plan-intro', type=float, default=1.2)
    ap.add_argument('--plan-iter', type=float, default=2.4)
    ap.add_argument('--plan-forecast', type=float, default=1.8)
    ap.add_argument('--plan-select', type=float, default=2.2)
    ap.add_argument('--result', type=float, default=2.8)
    ap.add_argument('--outro', type=float, default=4.)
    ap.add_argument('--cam-target', type=float, nargs=3, default=(.52, -.10, .79))
    ap.add_argument('--cam-distance', type=float, default=1.95)
    ap.add_argument('--cam-azimuth', type=float, default=-16., help='deg; 0 = looking along +x, from behind the arms')
    ap.add_argument('--cam-elevation', type=float, default=55.)
    ap.add_argument('--cam-orbit', type=float, default=7., help='deg of slow azimuth sway')
    ap.add_argument('--cam-period', type=float, default=48.)
    ap.add_argument('--cam-lens', type=float, default=46.)
    ap.add_argument('--cam-zoom', type=float, default=.66, help='distance factor while a cycle plans')
    ap.add_argument('--cam-focus', type=float, default=.7, help='how far the target moves toward the candidates')
    ap.add_argument('--ghost-cam-zoom', type=float, default=.9, help='--cam-zoom when ghost arms are drawn')
    ap.add_argument('--ghost-cam-focus', type=float, default=.3, help='--cam-focus when ghost arms are drawn')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    run = Path(a.run).resolve()
    bundle = Path(a.bundle).resolve() if a.bundle else run / 'blender'
    if a.rebuild_phases:
        rebuild_phases(run, Path(a.rebuild_phases).resolve())
        return 0
    build(run, bundle, a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
