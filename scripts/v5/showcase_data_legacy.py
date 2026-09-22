#!/usr/bin/env python3
"""The EARLIER real campaign (`scripts/mppi/run_real_mppi.py`, `outputs/mppi_real/run_*`) in the shape
`showcase_data.load` returns, so the Blender MPPI video renders it with the same three stages. Read-only.

Those runs are the THIN hose -- 11 mm radius, `film/calibration.json` `cable_radius_m` -- routed against a
CAPTURED SHAPE goal with a side/winding contract over four 20 mm pegs (`planning/captured_shape.py`), not the
peg-side objective `run_real_v5` scores. Six things differ from a v5 run directory, and each is read here:

  v5                                    this campaign
  ----------------------------------    ------------------------------------------------------------------
  config/<task>.json, config/rig.yaml   nothing; the pegs and the goal are in metrics.json + trajectory.npz
  observations/<tag>.npz                the state the planner held is the film's own phase header (`pre`);
                                        the settled state after a phase is the NEXT `trajectory['states']`
                                        row -- the probes come first in that array, so a cycle's own index
                                        is `len(probes) + k` (checked against the logged errors, below)
  metrics cycles[].plan_file            phases/phase_{k:03d}.json, by cycle order (checked: its `meta.action`
                                        is cycles[k]['action'] in all 29 cycles of run_20260911_212456)
  samples: valid/refused/pred per row   costs (1024,), actions (1024, H, 21) and the rollouts of the 64
                                        LOWEST-COST rows only (`best_rollouts[m]` is `argsort(costs)[m]`,
                                        run_real_mppi.py:1266). A row at cost >= 1e3 is one whose forecast
                                        diverged; `goal.py:262` floors those so they can never win. There is
                                        no recorded feasibility screen: reach and grasp failures are
                                        penalties inside the cost, not refusals.
  routing cost (peg sides)              mean shape error in mm (`error_before_mm` / `error_mm`)
  events.jsonl, run.log                 neither; the film's capture clock is the only clock

`check(data)` recomputes every cycle's logged `shape_after` -- mean/max/tip error, peg sides, winding -- from
the contract rebuilt here and raises unless all of them match, the same gate `showcase_data.check` is for a v5
run. On run_20260911_212456 all 29 cycles match, and the action -> keyframe rebuild in
`mppi_viz_data.LegacyRigPlanner` reproduces all 29 executed phase JSONs to 0.000 mm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mppi_paths import HOSE_ROUTING_ROOT

import numpy as np

ROOT = HOSE_ROUTING_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIVERGED = 1e3                  # planning/goal.py:262 floors a diverged rollout's cost here
LEGACY_KINDS = ('real two-hand learned MPPI',)


def is_legacy(run) -> bool:
    """Is this directory from the earlier campaign? (its metrics say so; a v5 run has config/rig.yaml)"""
    run = Path(run)
    path = run / 'metrics.json'
    if not path.exists():
        return False
    kind = str(json.loads(path.read_text()).get('kind') or '')
    return kind in LEGACY_KINDS or (not (run / 'config' / 'rig.yaml').exists() and (run / 'record').is_dir())


def film_head(run: Path, tag: str) -> dict:
    """The phase header the film writer put first in `film/<tag>/frames.jsonl` (its action and `pre` state)."""
    path = Path(run) / 'film' / tag / 'frames.jsonl'
    if not path.exists():
        return {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get('kind') == 'phase':
                return row
            break
    return {}


def action_vector(action) -> np.ndarray:
    """One recorded action dict -> the planner's 20 numbers, the layout `showcase_data.hand_paths` decodes.

    `PhaseAction.to_vector` is [s, sa, deltas (3x3), deltas_a (3x3), slack] (act/action.py:57-63); the first
    20 of those are exactly the v5 phase vector, and `slack` is 0 in every recorded row of this campaign.
    """
    a = dict(action)
    return np.concatenate([[float(a['s']), float(a['sa'])],
                           np.asarray(a['deltas'], np.float64).reshape(-1),
                           np.asarray(a['deltas_a'], np.float64).reshape(-1)])


def action_key(action) -> tuple:
    """A recorded action as an exact, hashable key (the numbers are stored, not recomputed)."""
    a = dict(action)
    return (int(a['role']), repr(float(a['s'])), repr(float(a['sa'])),
            tuple(repr(float(v)) for v in np.asarray(a['deltas'], np.float64).reshape(-1)),
            tuple(repr(float(v)) for v in np.asarray(a['deltas_a'], np.float64).reshape(-1)))


def phase_files(run, metrics=None) -> dict:
    """tag -> the phase JSON the rig was actually given for it, paired BY ACTION rather than by position.

    The executor numbered `phases/phase_NNN.json` as it wrote them, so a phase the rig never got to write
    shifts every later file: on run_20260911_222321 one probe has neither a file nor a film, and pairing by
    count puts each cycle's keyframes one phase early. Every phase carries its action in `meta.action` and
    every probe and cycle carries the same action in the record, so the pairing below is exact; a tag with no
    matching file simply has no plan, and both the video and the arm tracks leave it out.
    """
    run = Path(run)
    metrics = metrics if metrics is not None else json.loads((run / 'metrics.json').read_text())
    by_action = {}
    for path in sorted(run.glob('phases/phase_*.json')):
        plan = json.loads(path.read_text())
        act = (plan.get('meta') or {}).get('action')
        if act:
            by_action.setdefault(action_key(act), []).append(path)
    out = {}
    for i, row in enumerate(metrics.get('probes') or []):
        tag = 'probe_%03d' % int(row.get('k', i + 1))
        act = row.get('action') or (film_head(run, tag).get('action') or None)
        if act:
            paths = by_action.get(action_key(act)) or []
            if paths:
                out[tag] = paths.pop(0)
    for row in metrics.get('cycles') or []:
        tag = 'cycle_%03d' % int(row['cycle'])
        paths = by_action.get(action_key(row['action'])) or []
        if paths:
            out[tag] = paths.pop(0)
    return out


def samples(run: Path, index: int):
    """One cycle's candidates, in the v5 sample dict's shape. `index` is 0-based (the file's own numbering).

    `pred` is NaN for every row whose rollout the recorder did not keep: only the 64 lowest-cost rows of each
    iteration have a forecast, so anything drawn from a forecast must mask on `np.isfinite`.
    """
    rows = []
    for path in sorted(Path(run).glob('samples/cycle_%03d_iter_*.npz' % index)):
        z = np.load(path, allow_pickle=True)
        costs = np.asarray(z['costs'], np.float64)
        actions = np.asarray(z['actions'], np.float64)[:, :, :20]
        n = len(costs)
        best = np.asarray(z['best_rollouts'], np.float64) if 'best_rollouts' in z.files else np.zeros((0,))
        pred = np.full((n,) + best.shape[1:], np.nan) if len(best) else np.full(actions.shape[:2] + (0, 3), np.nan)
        if len(best):
            pred[np.argsort(costs, kind='stable')[:len(best)]] = best
        rows.append(dict(iteration=int(path.stem.rsplit('_', 1)[-1]) * np.ones(n, int),
                         state=np.asarray(z['state'], np.float64), actions=actions, costs=costs,
                         weights=np.asarray(z['weights'], np.float64),
                         valid=costs < DIVERGED, pred=pred,
                         roles=np.asarray(z['roles'], int),
                         family=np.full(n, 'sample'),
                         refused=np.where(costs < DIVERGED, '', 'forecast diverged')))
    if not rows:
        return None
    out = dict(state=rows[0]['state'], iterations=len(rows))
    for key in ('iteration', 'actions', 'costs', 'weights', 'valid', 'pred', 'roles', 'family', 'refused'):
        out[key] = np.concatenate([r[key] for r in rows])
    return out


class ShapeContractEvaluator:
    """`CapturedShapeContract` behind the interface `showcase_data.evaluator` returns for a v5 run.

    `evaluate(batch)` gives the same keys the video reads: `sides` (one per landmark peg, the contract's own
    side test), `side_gap_m`, and `cost` = the MEAN SHAPE ERROR in metres, which is the number this campaign
    reported per cycle (`error_mm`). It is not the planner's objective -- that is `GoalCost.cost`, whose
    per-candidate values are in the sample files and are what the video colours candidates by.
    """

    def __init__(self, contract):
        self.contract = contract
        self.landmarks = contract.landmarks

    def evaluate(self, batch) -> dict:
        g = self.contract.geometry(np.asarray(batch, np.float64))
        return dict(cost=np.asarray(g['mean'], np.float64), sides=np.asarray(g['sides'], bool),
                    side_gap_m=np.asarray(g['side_gap'], np.float64),
                    winding=np.asarray(g['winding'], int), mean_m=np.asarray(g['mean'], np.float64),
                    max_m=np.asarray(g['maximum'], np.float64), tip_m=np.asarray(g['tip'], np.float64))


def evaluator(data) -> ShapeContractEvaluator:
    """The run's own contract: its captured target, its measured pegs, its table."""
    from hose_routing.planning.captured_shape import CapturedShapeContract
    return ShapeContractEvaluator(CapturedShapeContract(data['target'], data['metrics']['pegs'],
                                                        table_z=data['table_z']))


def landmarks(data, ev) -> list:
    """The contract's landmark pegs for the render: which peg, the goal-side normal, and where on the goal."""
    target = np.asarray(data['target'], np.float64)
    out = []
    for lm in ev.landmarks:
        j = int(lm['segment'])
        out.append(dict(peg=int(lm['peg_id']), point=[float(v) for v in target[min(j + 1, len(target) - 1)]],
                        normal=[float(v) for v in np.asarray(lm['normal'], np.float64)],
                        behind=bool(float(lm['normal'][0]) > 0)))
    return out


def load(run) -> dict:
    """A legacy run directory -> the dict the video stages read (the keys `showcase_data.load` returns)."""
    run = Path(run)
    metrics = json.loads((run / 'metrics.json').read_text())
    traj = np.load(run / 'trajectory.npz', allow_pickle=True)
    states = np.asarray(traj['states'], np.float64)
    goal = np.asarray(traj['goal'], np.float64)
    cal = json.loads((run / 'film' / 'calibration.json').read_text()) \
        if (run / 'film' / 'calibration.json').exists() else {}
    radius = float(cal.get('cable_radius_m', .011))
    table_z = float(cal.get('table_z', .75))
    pegs = [dict(x=float(p['x']), y=float(p['y']), r=float(p.get('radius', p.get('r', .0057))),
                 h=float(p.get('height', p.get('h', .02)))) for p in metrics['pegs']]

    # The executor numbered its phase files as it ran: the probes first, then the cycles
    # (`act/executor.py` `self.n`), which is the only link between a cycle and the keyframes it was given.
    probe_rows = list(metrics.get('probes') or [])
    cycle_rows = list(metrics.get('cycles') or [])
    tags = ['probe_%03d' % int(r.get('k', i + 1)) for i, r in enumerate(probe_rows)] + \
           ['cycle_%03d' % int(r['cycle']) for r in cycle_rows]
    heads = [film_head(run, t) for t in tags]
    pres = [np.asarray(h['pre'], np.float64) if h.get('pre') is not None else None for h in heads]

    plans = phase_files(run, metrics)

    def plan_of(tag):
        path = plans.get(tag)
        return json.loads(path.read_text()) if path is not None else None

    probes = []
    for i, row in enumerate(probe_rows):
        pre = pres[i] if pres[i] is not None else states[0]
        post = states[i + 1] if i + 1 < len(states) else None
        action = (heads[i].get('action') or {})
        executed = action_vector(action) if action else None
        probes.append(dict(n=int(row.get('k', i + 1)), row=row, pre=pre, post=post, plan=plan_of(tags[i]),
                           obs=dict(pre=pre, post=post, settled=None, planned=executed,
                                    executed=executed, meta={})))

    cycles = []
    for k, row in enumerate(cycle_rows):
        n = int(row['cycle'])
        i = len(probe_rows) + k
        pre = pres[i] if pres[i] is not None else states[i]
        post = states[i + 1] if i + 1 < len(states) else None          # the settled state the record scores
        executed = action_vector(row['action'])
        shape = row.get('shape_after') or {}
        cycles.append(dict(
            n=n, row=_row(row), pre=pre, post=post,
            # the record keeps the COMMANDED action only (what the rig reached is not logged as an action),
            # so `planned` and `executed` are the same vector; the ghost arms use the phase JSON regardless
            obs=dict(pre=pre, post=post, settled=None, planned=executed, executed=executed, meta={}),
            samples=samples(run, k), predicted=np.zeros((0, len(pre), 3)), plan=plan_of(tags[i]),
            family='committed action', plan_index=i, shape_after=shape))

    final = states[-1]
    return dict(run=run, metrics=metrics, task=dict(task_id=Path(str(metrics['args'].get('goal_reference')
                                                                    or 'captured goal')).stem),
                task_live={}, cycles=cycles, probes=probes, pegs=pegs, table_z=table_z, radius=radius,
                goal=np.asarray(traj['physical_goal_line'], np.float64) if 'physical_goal_line' in traj.files
                else goal,
                target=goal, states=states, start=states[0], final=final,
                success=bool(metrics.get('success')),
                final_cost=float((metrics.get('final_error_mm') if metrics.get('final_error_mm') is not None
                                  else np.nan)),
                initial_cost=float(metrics.get('initial_error_mm', np.nan)),
                legacy=True, cost_label='shape error', cost_unit='mm',
                shape_final=metrics.get('shape_current') or {})


def _row(row: dict) -> dict:
    """One recorded cycle, with the three numbers the video's captions read under their v5 names.

    They are MILLIMETRES of mean shape error here, not a routing cost; `data['cost_label']` says so and every
    caption and README line that prints them uses it.
    """
    out = dict(row)
    out['cost_before'] = float(row.get('error_before_mm', np.nan))
    out['final_cost'] = float(row.get('error_mm', np.nan))
    out['predicted_cost'] = float(row.get('predicted_error_mm', np.nan))
    out['planning'] = dict(predicted_states=[], choice='', planning_seconds=row.get('plan_s'))
    out['selected_family'] = 'committed action'
    return out


def check(data, tol=.05) -> ShapeContractEvaluator:
    """Recompute every cycle's logged `shape_after` from its settled state. Raises if the contract differs.

    Mean/max/tip error to `tol` mm, and the peg sides and winding exactly -- the whole logged report, so the
    sides this video draws are the sides the run itself recorded.
    """
    ev = evaluator(data)
    bad, checked = [], 0
    for c in data['cycles']:
        want = c.get('shape_after') or {}
        if c['post'] is None or not want:
            continue
        got = ev.contract.report(np.asarray(c['post'])[None])
        checked += 1
        for key in ('mean_error_mm', 'max_error_mm', 'tip_error_mm'):
            if key in want and abs(float(got[key]) - float(want[key])) > tol:
                bad.append('cycle %d %s: %.3f vs %.3f' % (c['n'], key, got[key], want[key]))
        for key in ('side_matches', 'winding_difference'):
            if key in want and list(got[key]) != list(want[key]):
                bad.append('cycle %d %s: %s vs %s' % (c['n'], key, got[key], want[key]))
    if bad:
        raise AssertionError('the rebuilt contract does not match the record: ' + '; '.join(bad[:6]))
    data['cost_axis_checked'] = checked
    return ev


def _film_size(run: Path):
    """The recorded film's own frame size, read off the first frame that exists."""
    for path in sorted(Path(run).glob('film/*/f00000.jpg')):
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
        except ImportError:
            return None
    return None


def camera(run) -> dict:
    """The overhead camera as this campaign recorded it: pose in `film/calibration.json`, K on every frame.

    The film writer stored the intrinsics per frame (`K` = [fx, fy, cx, cy] at the frame's own `scale`), so
    they are read from the first frame of the first phase and scaled to that frame's size.
    """
    run = Path(run)
    cal = json.loads((run / 'film' / 'calibration.json').read_text())
    T = np.asarray(cal['T_world_overhead'], np.float64).reshape(4, 4)
    size = _film_size(run) or (int(cal.get('width', 960)), int(round(cal.get('width', 960) * 9 / 16)))
    K = np.eye(3)
    for path in sorted(run.glob('film/*/frames.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'K' in row:
                fx, fy, cx, cy = (float(v) for v in row['K'])
                scale = float(row.get('scale', 1.))
                K = np.array([[fx * scale, 0., cx * scale], [0., fy * scale, cy * scale], [0., 0., 1.]])
                break
        if K[0, 0] != 1.:
            break
    T_inv = np.linalg.inv(T)

    def project(points):
        p = np.asarray(points, np.float64).reshape(-1, 3)
        cam = (T_inv[:3, :3] @ p.T).T + T_inv[:3, 3]
        uv = (K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]

    return dict(K=K, T_world_overhead=T, size=size, project=project)


if __name__ == '__main__':
    data = load(sys.argv[1] if len(sys.argv) > 1 else 'outputs/mppi_real/run_20260911_212456')
    print('%d cycles, %d mm hose, %d pegs %d mm tall, shape error %.0f -> %.0f mm, success %s'
          % (len(data['cycles']), round(2000 * data['radius']), len(data['pegs']),
             round(1000 * data['pegs'][0]['h']), data['initial_cost'], data['final_cost'], data['success']))
    ev = check(data)
    print("the run's own contract reproduces every logged cycle report (%d cycles)" % data['cost_axis_checked'])
    for c in data['cycles'][:5]:
        s = c['samples']
        print('cycle %2d: %6.1f -> %6.1f mm (forecast %6.1f) | %d samples, %d with a forecast, %d diverged'
              % (c['n'], c['row']['cost_before'], c['row']['final_cost'], c['row']['predicted_cost'],
                 0 if s is None else len(s['costs']),
                 0 if s is None else int(np.isfinite(s['pred'][:, 0, 0, 0]).sum()),
                 0 if s is None else int((~s['valid']).sum())))
