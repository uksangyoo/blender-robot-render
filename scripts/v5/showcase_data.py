#!/usr/bin/env python3
"""What one real-rig routing run contains, in the terms a figure needs. Read-only, no rendering.

`load(run)` returns a dict of plain numpy/py objects assembled from a `run_real_v5` output directory:
the task (pegs, goal line), the tracked hose before and after every cycle, the plan that was chosen, and --
the part the first report left out -- the MPPI SAMPLES: every candidate action, its family, whether it was
refused and why, its cost, and the learned model's predicted hose for it.

The action encoding is the release's (`mind-cable-v5/sim/mppi_planner.decode_action`, DELTA_SCALE 0.12):
one action row is (phase, 20), and each phase's 20 numbers are

    [arc_hand0, arc_hand1, then 2 hands x 3 waypoints x (dx, dy, dz) / 0.12]

so `hand_paths` turns a row into the two commanded gripper polylines in world metres, which is what
"the sampled action" means on a figure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mppi_paths import HOSE_ROUTING_ROOT

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import showcase_data_legacy as legacy

DELTA_SCALE = .12                      # mind-cable-v5/sim/mppi_planner.DELTA_SCALE
N_WAYPOINTS = 3
FAMILY_WORDS = {'goal_carry': 'goal carry', 'crossing_carry': 'crossing carry',
                'bimanual_carry': 'two-hand carry', 'single_carry': 'one-hand carry',
                'tail_sweep': 'tail sweep', 'release': 'release sampler', 'mean': 'goal carry',
                'best_sample': 'best sample', 'null': 'hold still'}
REFUSAL_WORDS = {'': 'accepted', 'other_section': 'grasps the wrong section', 'projection': 'no feasible path',
                 'grasps_moved': 'grasp snapped elsewhere', 'crossing_stage': 'wrong crossing stage',
                 'peg': 'drives into a peg', 'reach': 'out of arm reach', 'stretch': 'stretches the hose',
                 'floor': 'below the table', 'self_contact': 'hits itself'}


def hand_paths(action_phase, state):
    """One phase's 20 numbers + the hose it acts on -> (nodes, paths): grasp node and world path per hand.

    paths[h] is (1 + N_WAYPOINTS, 3): the grasp point followed by the three commanded waypoints.
    """
    a = np.asarray(action_phase, np.float64)
    state = np.asarray(state, np.float64)
    last = len(state) - 1
    nodes = np.clip(np.rint(a[:2] * last).astype(int), 0, last)
    deltas = a[2:].reshape(2, N_WAYPOINTS, 3) * DELTA_SCALE
    paths = np.stack([np.concatenate([state[n][None], state[n] + deltas[h].cumsum(axis=0)])
                      for h, n in enumerate(nodes)])
    return nodes, paths


def moved(paths, min_m=.005):
    """Which of the two hands actually goes anywhere (the other is a still pin)."""
    return np.linalg.norm(paths[:, -1] - paths[:, 0], axis=-1) > min_m


def observation(run: Path, tag: str):
    """One stage's tracked hose, on the task's own 28 material points. -> dict or None

    `frames/*_nodes.npz` is the tracker's raw chain (37 points here) and CANNOT be scored against the
    goal; `observations/<tag>.npz` holds the resampled 28-node centres the runner itself planned on,
    the five settled observations the success test used, and the action as planned and as executed.
    """
    path = run / 'observations' / (tag + '.npz')
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    get = lambda k: np.asarray(z[k], np.float64) if k in z.files else None
    meta = {}
    if 'meta_json' in z.files:
        try:
            meta = json.loads(str(z['meta_json']))
        except (ValueError, TypeError):
            meta = {}
    return dict(pre=get('pre_centers'), post=get('post_centers'), settled=get('settled_centerlines'),
                planned=get('planned_mind_action'), executed=get('executed_mind_action'),
                phase_vector=get('phase_vector'), meta=meta)


def samples(run: Path, cycle: int):
    """Every candidate the planner drew for one cycle, with its iteration index. -> dict of arrays"""
    rows = []
    for path in sorted(run.glob('samples/cycle_%03d_iter_*.npz' % cycle)):
        z = np.load(path, allow_pickle=True)
        n = len(z['costs'])
        rows.append(dict(iteration=int(path.stem.rsplit('_', 1)[-1]) * np.ones(n, int),
                         state=np.asarray(z['state'], np.float64),
                         actions=np.asarray(z['actions'], np.float64),
                         costs=np.asarray(z['costs'], np.float64),
                         weights=np.asarray(z['weights'], np.float64),
                         valid=np.asarray(z['valid'], bool),
                         pred=np.asarray(z['pred'], np.float64),
                         roles=np.asarray(z['roles'], int),
                         family=np.asarray(z['family']).astype(str) if 'family' in z.files
                         else np.full(n, 'release'),
                         refused=np.asarray(z['refused']).astype(str) if 'refused' in z.files
                         else np.full(n, '')))
    if not rows:
        return None
    out = dict(state=rows[0]['state'], iterations=len(rows))
    for key in ('iteration', 'actions', 'costs', 'weights', 'valid', 'pred', 'roles', 'family', 'refused'):
        out[key] = np.concatenate([r[key] for r in rows])
    return out


def phase_keyframes(plan) -> dict:
    """An executed phase JSON -> the commanded gripper keyframes, per arm, as the rig was given them.

    Each keyframe is dict(name, position (world m), tangent (the jaw-opening axis, in the table plane),
    grip (1 open, 0 shut)). `rig_arm` maps the planner's sim roles (minus_y / plus_y) to the physical
    arms, and the two base transforms are the ones this run measured -- everything a renderer needs to
    put the arms where they really were.
    """
    meta = plan['meta']
    arms = {}
    for role in ('minus_y', 'plus_y'):
        arms[role] = [dict(name=str(k['name']), position=np.asarray(k['position'], np.float64),
                           tangent=np.asarray(k['tangent'], np.float64), grip=float(k['grip']))
                      for k in plan.get(role) or []]
    return dict(arms=arms, mover=str(plan.get('mover') or ''), rig_arm=dict(meta.get('rig_arm') or {}),
                carry=bool(meta.get('carry')), carry_lift=float(meta.get('carry_lift') or 0.),
                T_world_leftbase=np.asarray(meta['T_world_leftbase'], np.float64).reshape(4, 4),
                T_world_rightbase=np.asarray(meta['T_world_rightbase'], np.float64).reshape(4, 4),
                action=str(meta.get('action') or ''), mover_node=meta.get('mover_node'),
                anchor_node=meta.get('anchor_node'))


def task_file(run: Path, metrics: dict) -> Path:
    """The task JSON the run copied into `config/`.

    `metrics['task']` is a path string in the earlier v5 runs and a dict carrying its own `path` in the later
    ones (and in the sim campaigns, where the file is named for the goal -- `L1_far.json` -- not `G*.json`).
    The copy keeps the basename, so that is tried first; otherwise the task is the one file in `config/` that
    defines pegs.
    """
    spec = metrics.get('task')
    name = Path(str(spec.get('path') if isinstance(spec, dict) else (spec or ''))).name
    cfg = run / 'config'
    if name and (cfg / name).exists():
        return cfg / name
    for path in sorted(cfg.glob('*.json')):
        try:
            if 'pegs' in json.loads(path.read_text()):
                return path
        except (ValueError, OSError):
            continue
    raise FileNotFoundError('no task JSON with pegs in %s' % cfg)


def load(run) -> dict:
    """A run directory -> everything the showcase figures draw.

    A directory from the EARLIER real campaign (`run_real_mppi.py`, the thin hose against a captured shape
    goal) is read by `showcase_data_legacy`, which returns these same keys plus `legacy=True`.
    """
    run = Path(run)
    if legacy.is_legacy(run):
        return legacy.load(run)
    metrics = json.loads((run / 'metrics.json').read_text())
    task = json.loads(task_file(run, metrics).read_text())
    traj = np.load(run / 'trajectory.npz', allow_pickle=True)
    cycles = []
    for row in metrics.get('cycles') or []:
        n = int(row['cycle'])
        obs = observation(run, 'cycle_%03d' % n) or {}
        plan = run / 'phases' / Path(str(row.get('plan_file') or '')).name
        cycles.append(dict(
            n=n, row=row, pre=obs.get('pre'), post=obs.get('post'), obs=obs,
            samples=samples(run, n),
            predicted=np.asarray(row.get('planning', {}).get('predicted_states') or [], np.float64),
            plan=json.loads(plan.read_text()) if plan.name and plan.exists() else None,
            family=FAMILY_WORDS.get(str((row.get('planning') or {}).get('mean_family')
                                        if row.get('selected_family') == 'mean' else
                                        row.get('selected_family')), str(row.get('selected_family'))),
        ))
    probes = []
    for i, row in enumerate(metrics.get('probes') or []):
        obs = observation(run, 'probe_%03d' % (i + 1)) or {}
        probes.append(dict(n=i + 1, row=row, obs=obs, pre=obs.get('pre'), post=obs.get('post')))
    pegs = [dict(x=float(p['x']), y=float(p['y']), r=float(p.get('radius', p.get('r', .01))),
                 h=float(p.get('height', p.get('h', .045)))) for p in task['pegs']]
    task_live = dict(task)
    if metrics.get('planning_pegs'):
        task_live['pegs'] = metrics['planning_pegs']            # the pegs the run measured and planned on
        pegs = [dict(x=float(p['x']), y=float(p['y']), r=float(p.get('radius', .01)),
                     h=float(p.get('height', .045))) for p in metrics['planning_pegs']]
    return dict(run=run, metrics=metrics, task=task, task_live=task_live, cycles=cycles, probes=probes, pegs=pegs,
                table_z=float(task.get('table_z', .75)), radius=float(task.get('radius_m', .035)),
                goal=np.asarray(traj['physical_goal_line'], np.float64) if 'physical_goal_line' in traj.files
                else np.asarray(task['target_line'], np.float64),
                target=np.asarray(traj['target'], np.float64),
                states=np.asarray(traj['states'], np.float64),
                start=np.asarray(traj['states'][0], np.float64),
                final=np.asarray(traj['states'][-1], np.float64),
                success=bool(metrics.get('success')),
                final_cost=float(metrics.get('final_cost', float('nan'))),
                initial_cost=float(metrics.get('initial_cost', float('nan'))))


def with_extent(data):
    """`load` plus the shared panel limits (imported here so the data module stays plot-free)."""
    import showcase_style as S
    data['extent'] = S.extent(data)
    return data


def camera(run) -> dict:
    """The overhead camera as the run recorded it, plus a world -> pixel projection.

    The intrinsics come from `calibration.json` (the run's own, not a stored calibration), the pose from
    the same file's `T_world_overhead`. Distortion is all zeros there and the runner deprojects without
    undistorting, so a plain pinhole projection matches what the detector and tracker saw.
    """
    if legacy.is_legacy(run):
        return legacy.camera(run)
    cal = json.loads((Path(run) / 'calibration.json').read_text())
    K = np.asarray(cal['camera']['K_row_major'], np.float64).reshape(3, 3)
    T_inv = np.linalg.inv(np.asarray(cal['transforms']['T_world_overhead'], np.float64).reshape(4, 4))

    def project(points):
        p = np.asarray(points, np.float64).reshape(-1, 3)
        cam = (T_inv[:3, :3] @ p.T).T + T_inv[:3, 3]
        uv = (K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]

    return dict(K=K, T_world_overhead=np.asarray(cal['transforms']['T_world_overhead'],
                                                 np.float64).reshape(4, 4),
                size=(int(cal['camera']['width']), int(cal['camera']['height'])), project=project)


def evaluator(data):
    """The run's OWN routing objective, rebuilt so predictions can be scored on the figure's axis.

    Same class and same two settings the run logged (`metrics['objective']`: near_m 0.07, top allowance
    0.0205), on the measured pegs -- so a recomputed cost is comparable with `routing.cost` in the record,
    which `check(data)` asserts before any figure uses it.
    """
    import sys
    from types import SimpleNamespace
    root = HOSE_ROUTING_ROOT
    for path in (root, root / 'mind-cable-v5'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from hose_routing.v5.routing_objective import PegSideEvaluator
    from hose_routing.v5 import goal_suite as GS
    contract, target = GS.goal_contract(data['task_live'])
    spec = data['metrics'].get('objective') or {}
    return PegSideEvaluator(SimpleNamespace(contract=contract, target=np.asarray(target, np.float64)),
                            near_m=float(spec.get('near_m', .07)),
                            top_allowance_m=float(spec.get('top_allowance_m', .0205)))


def score_predictions(ev, pred):
    """(K, N, 3) predicted hose states -> dict(cost (K,), sides (K, P), gap_mm (K, P)) on the run's axis."""
    pred = np.asarray(pred, np.float64)
    out = ev.evaluate(pred)
    return dict(cost=np.asarray(out['cost'], np.float64), sides=np.asarray(out['sides'], bool),
                gap_mm=1e3 * np.asarray(out['side_gap_m'], np.float64),
                routed=np.asarray(out['sides'], bool).all(axis=1))


def check(data, tol=.02):
    """Recompute every cycle's logged cost from its observation. Raises if the axis does not match.

    Only runs whose objective is `peg_sides` log a per-cycle `routing` block to compare against; a run
    scored on an earlier objective (`release_captured_contract`) has no such number, so there is nothing
    to reproduce. The evaluator is still the right peg-side readout for its board, so it is returned with
    `data['cost_axis_checked']` recording how many cycles were verified.
    """
    if data.get('legacy'):
        return legacy.check(data)
    ev = evaluator(data)
    bad, checked = [], 0
    for c in data['cycles']:
        logged = (c['row'].get('routing') or {}).get('cost')
        if c['post'] is None or logged is None:
            continue
        mine = float(ev.evaluate(np.asarray(c['post'])[None])['cost'][0])
        checked += 1
        if abs(mine - float(logged)) > tol:
            bad.append('cycle %d: %.3f vs %.3f' % (c['n'], mine, float(logged)))
    if bad:
        raise AssertionError('recomputed routing cost does not match the record: ' + '; '.join(bad))
    data['cost_axis_checked'] = checked
    return ev


if __name__ == '__main__':
    import sys
    data = load(sys.argv[1] if len(sys.argv) > 1 else 'outputs/mppi_real_v5/run_20260917_150620')
    print('task %s, %d pegs, hose radius %.3f m' % (data['task'].get('task_id'), len(data['pegs']), data['radius']))
    print('cost %.2f -> %.2f, success %s' % (data['initial_cost'], data['final_cost'], data['success']))
    ev = check(data)
    print('the run\'s own evaluator reproduces every logged cycle cost')
    for c in data['cycles']:
        s = c['samples']
        nodes, paths = hand_paths(np.asarray(c['obs']['executed'], np.float64), c['pre'])
        scored = None if s is None else score_predictions(ev, s['pred'][:, -1])
        print('cycle %d: %-12s cost %.2f -> %.2f | %d samples (%d valid) | forecast cost min %.2f med %.2f'
              ' | grasps %s moving %s'
              % (c['n'], c['family'], c['row']['cost_before'], c['row']['final_cost'],
                 0 if s is None else len(s['costs']), 0 if s is None else int(s['valid'].sum()),
                 scored['cost'][s['valid']].min() if scored is not None else float('nan'),
                 np.median(scored['cost'][s['valid']]) if scored is not None else float('nan'),
                 nodes.tolist(), moved(paths).tolist()))
