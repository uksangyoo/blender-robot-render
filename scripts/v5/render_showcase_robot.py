#!/usr/bin/env python3
"""The rig itself, rendered: one executed phase -> IK for both arms -> white publication stills.

READ-ONLY on the hardware. It solves the rig's own IK (pyroki, the same call `execute_phase_yam` makes),
composes a MuJoCo world with both arms at the base poses this run measured, and renders offscreen with
EGL. No CAN bus is opened; the i2rt driver is never imported; nothing moves.

    cd ~/Projects/hose-routing/mild-trackdlo/yam_bimanual
    .venv_trace/bin/python ~/Projects/hose-routing/scripts/v5/render_showcase_robot.py \
        --run ~/Projects/hose-routing/outputs/mppi_real_v5/run_<stamp> --verify

Writes `<run>/showcase/robot/*.png` and a `manifest.json` the figure script reads, so the rendering
(which needs this interpreter) and the composing (which needs .venv-v5) stay apart.

`--verify` runs the whole-chain check first: where does `grasp_site` actually END UP in the composed
world, against the position the phase JSON commanded. That is the one test an IK report cannot make --
it fails if the base poses, the role-to-arm mapping, the IK or the model composition disagree.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from mppi_paths import HOSE_ROUTING_ROOT, RENDER_REPO, SCRIPTS

YAM = os.path.expanduser(os.environ.get('YAM_BIMANUAL_DIR') or str(HOSE_ROUTING_ROOT / 'mild-trackdlo/yam_bimanual'))
HR = str(HOSE_ROUTING_ROOT)
for path in (YAM, os.path.join(YAM, 'scripts'), os.path.join(HR, 'scripts'),
             HR, str(SCRIPTS)):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import mujoco                                                                 # noqa: E402
import showcase_scene as SC                                                   # noqa: E402
import trace_board as tb                                                      # noqa: E402
import trace_board_pyroki as tp                                               # noqa: E402
import execute_phase_yam as ex                                                # noqa: E402

TAG = {'left': 'L', 'right': 'R'}
# The keyframes worth a figure. `approach` and `retreat` are `grasp`/`place` +- 50 mm in z and
# look identical; `close` is the contact moment; `carry*` is the only time the arms differ.
STORY = ('grasp', 'close', 'lift', 'carry1', 'carry2', 'place', 'open')


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ t
    return out


def solve_phase(phase, config):
    """One phase JSON -> per sim role: the physical arm, the keyframe names, q6, grip, IK residual.

    `meta.rig_arm` is read from the plan, never guessed: `execute_phase_yam` explains that assuming
    otherwise drives each arm to the other's pose. `ex.build_waypoints` takes five arguments and picks
    one wrist-roll sign for the whole phase (per keyframe made the gripper spin with the jaws shut).
    """
    meta = phase['meta']
    cfg = tb.load_config(config)
    chans = {t: cfg['arms'][t]['channel'] for t in ('left', 'right')}
    robot, mjcf = tp.build_pyroki_robot(), tp.MjcfSite()
    rc = {'left': tp.load_robot_config(os.path.join(tp.TELEOP, 'robot_configs/yam/left.yaml'),
                                       chans['left']),
          'right': tp.load_robot_config(tp.DEFAULT_ROBOT_CONFIG, chans['right'])}
    limits = {t: tp.config_joint_limits(rc[t]) for t in ('left', 'right')}
    world = {'left': np.asarray(meta['T_world_leftbase'], float).reshape(4, 4),
             'right': np.asarray(meta['T_world_rightbase'], float).reshape(4, 4)}
    out = {}
    for role in ('minus_y', 'plus_y'):
        arm = meta['rig_arm'][role]
        w = ex.build_waypoints(phase[role], inv(world[arm]), robot, mjcf, limits[arm])
        out[role] = dict(rig_arm=arm, tag=TAG[arm], names=[x['name'] for x in w],
                         q6=np.array([x['q6'] for x in w], float),
                         grip=np.array([x['grip'] for x in w], float),
                         pos_err_mm=np.array([x['pos_err_mm'] for x in w], float),
                         commanded=np.array([k['position'] for k in phase[role]], float),
                         ok=[bool(x['ok'] and x['limit_ok']) for x in w])
    return out, meta, world


def paired(sol, wanted=STORY):
    """The two arms' keyframes lined up by name, for the named story frames. -> [(label, {role: row})]

    The arms have different keyframe counts (11 vs 7 here), so a frame index means nothing across them.
    A frame takes each arm's own keyframe of that name when it has one, and otherwise the arm's last
    keyframe at or before it -- which is what the rig does: the short arm holds while the long one drives.
    """
    frames = []
    for name in wanted:
        row = {}
        for role, s in sol.items():
            names = list(s['names'])
            if name in names:
                i = names.index(name)
            else:
                order = {n: j for j, n in enumerate(wanted)}
                seen = [j for j, n in enumerate(names) if order.get(n, -1) <= order.get(name, 0)]
                i = seen[-1] if seen else 0
            row[role] = dict(name=names[i], q6=s['q6'][i], grip=float(s['grip'][i]), tag=s['tag'])
        if any(r['name'] == name for r in row.values()):
            frames.append((name, row))
    return frames


def verify(model, data, qadr, gadr, sol, *, log=print):
    """Where does grasp_site actually land, against what the phase commanded? -> worst error [m]"""
    worst = 0.
    for role, s in sol.items():
        site = '%s_grasp_site' % s['tag']
        for i, name in enumerate(s['names']):
            q = {s['tag']: s['q6'][i]}
            g = {s['tag']: s['grip'][i]}
            for other in sol.values():
                if other['tag'] != s['tag']:
                    q.setdefault(other['tag'], np.zeros(6))
                    g.setdefault(other['tag'], 1.)
            SC.pose(model, data, qadr, gadr, q, g)
            got = np.asarray(data.site(site).xpos, float)
            err = float(np.linalg.norm(got - s['commanded'][i]))
            worst = max(worst, err)
    log('whole-chain check: worst grasp_site vs commanded %.3f mm over %d keyframes'
        % (1e3 * worst, sum(len(s['names']) for s in sol.values())))
    return worst


def fan_tubes(run, cycle, metrics, task, *, count=48, radius=.009):
    """The planner's own candidate forecasts, as thin tubes coloured by forecast cost.

    Scored with the run's own evaluator so the colour means the same thing as it does in the plots:
    deep teal = the model expects every peg side correct, pale grey = it does not. Only feasible
    candidates are drawn, spread evenly across the cost range so the fan shows the spread rather than
    48 near-copies of the best one.
    """
    import glob
    sys.path.insert(0, str(SCRIPTS))
    import showcase_data as D
    import showcase_style as S
    from matplotlib.colors import Normalize
    data = D.load(run)
    ev = D.evaluator(data)
    rows = []
    for path in sorted(glob.glob(os.path.join(run, 'samples', 'cycle_%03d_iter_*.npz' % cycle))):
        z = np.load(path, allow_pickle=True)
        pred, valid = np.asarray(z['pred'], float)[:, -1], np.asarray(z['valid'], bool)
        cost = D.score_predictions(ev, pred)['cost']
        for k in np.flatnonzero(valid):
            rows.append((float(cost[k]), pred[k]))
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])
    pick = np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))
    cmap = S.candidate_cmap()
    norm = Normalize(vmin=0, vmax=max(1e-6, rows[pick[-1]][0]))
    tubes = []
    for i in pick[::-1]:                     # worst first, so the best tubes are added last
        cost, points = rows[i]
        r, g, b, _ = cmap(norm(cost))
        tubes.append(dict(points=SC.polyline(points), radius=radius,
                          rgba='%.3f %.3f %.3f 1' % (r, g, b)))
    print('fan: %d candidate forecasts, cost %.2f to %.2f' % (len(tubes), rows[pick[0]][0],
                                                              rows[pick[-1]][0]))
    return tubes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--cycle', type=int, default=3)
    ap.add_argument('--config', default=os.path.join(YAM, 'config.yaml'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--width', type=int, default=2600)
    ap.add_argument('--height', type=int, default=1625)
    ap.add_argument('--views', nargs='+', default=['hero'])
    ap.add_argument('--sweep', action='store_true', help='render a camera sweep and stop')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--ghost-alpha', type=float, default=.62)
    ap.add_argument('--forecast-radius', type=float, default=.009,
                    help='the forecast tube [m]; at the hose radius it reads as a second hose')
    ap.add_argument('--fan', type=int, default=48, help='candidate forecasts to draw in the fan render')
    a = ap.parse_args(argv)

    run = a.run
    out = a.out or os.path.join(run, 'showcase', 'robot')
    os.makedirs(out, exist_ok=True)
    metrics = json.loads(open(os.path.join(run, 'metrics.json')).read())
    task = json.loads(open(next(
        os.path.join(run, 'config', f) for f in sorted(os.listdir(os.path.join(run, 'config')))
        if f.startswith('G') and f.endswith('.json'))).read())
    pegs = metrics.get('planning_pegs') or task['pegs']
    row = next(c for c in metrics['cycles'] if int(c['cycle']) == a.cycle)
    phase_name = os.path.basename(row['plan_file'])
    phase = json.loads(open(os.path.join(run, 'phases', phase_name)).read())
    obs = np.load(os.path.join(run, 'observations', 'cycle_%03d.npz' % a.cycle), allow_pickle=True)
    pre = np.asarray(obs['pre_polyline'], float)
    post = SC.polyline(np.asarray(obs['post_centers'], float))
    forecast = np.asarray(row['planning']['predicted_states'], float)[-1]

    t0 = time.time()
    sol, meta, world = solve_phase(phase, a.config)
    print('IK %.2f s for %s (cycle %d)' % (time.time() - t0, phase_name, a.cycle))
    for role, s in sol.items():
        print('  %-7s = rig %-5s  %2d keyframes  worst IK residual %.2f mm  all accepted %s'
              % (role, s['rig_arm'], len(s['names']), s['pos_err_mm'].max(), all(s['ok'])))

    def world_model(hose, tubes=()):
        model, data, qadr, gadr = SC.build(
            pegs, hose, world['left'], world['right'], width=a.width, height=a.height,
            hose_radius=float(task['radius_m']), table_z=float(task['table_z']), tubes=tubes)
        return model, data, qadr, gadr

    def rest(sol):
        """The arms where the rig actually leaves them (the phase's last keyframe), not at zero."""
        q = {s['tag']: s['q6'][-1] for s in sol.values()}
        g = {s['tag']: float(s['grip'][-1]) for s in sol.values()}
        return q, g

    forecast_tube = dict(points=SC.polyline(forecast), rgba=SC.GHOST_RGBA, radius=a.forecast_radius)
    model, data, qadr, gadr = world_model(pre, tubes=[forecast_tube])
    renderer = mujoco.Renderer(model, a.height, a.width)
    SC.render_flags(renderer)
    opt = SC.opts()
    import imageio.v2 as iio
    written = {}

    # always run the whole-chain check: it is ~0.1 s and the figure quotes its result
    chain_worst_mm = 1e3 * verify(model, data, qadr, gadr, sol,
                                  log=print if a.verify else (lambda *_: None))

    if a.sweep:
        for az in (152, 180, 208, 232):
            for el in (-26, -34, -42, -52):
                cam = SC.free_cam(model, [.44, -.10, .80], 1.8, az, el)
                SC.pose(model, data, qadr, gadr,
                        {t: np.zeros(6) for t in ('L', 'R')}, {t: 1. for t in ('L', 'R')})
                renderer.update_scene(data, camera=cam, scene_option=opt)
                name = 'sweep_az%03d_el%03d.png' % (az, -el)
                iio.imwrite(os.path.join(out, name), renderer.render())
                print('  ' + name)
        renderer.close()
        return 0

    frames = paired(sol)
    print('story frames: %s' % ', '.join(n for n, _ in frames))
    for view in a.views:
        cam = SC.view(model, view)
        for label, row_frames in frames:
            q = {r['tag']: r['q6'] for r in row_frames.values()}
            g = {r['tag']: r['grip'] for r in row_frames.values()}
            SC.pose(model, data, qadr, gadr, q, g)
            img, _ = SC.composite_ghost(renderer, data, cam, opt, alpha=a.ghost_alpha)
            name = 'cycle%02d_%s_%s.png' % (a.cycle, view, label)
            iio.imwrite(os.path.join(out, name), img)
            written['%s/%s' % (view, label)] = name
            print('  %-28s %s' % (label, name))
    renderer.close()
    del renderer

    # the states the figure wants with the arms parked where the rig leaves them:
    # the hose before the move, the hose after it, and the planner's FAN of candidate forecasts
    fan = []
    if a.fan > 0:
        fan = fan_tubes(run, a.cycle, metrics, task, count=a.fan, radius=a.forecast_radius)
    for tag, hose, tubes in (('before', pre, []), ('after', post, []),
                             ('fan', pre, fan), ('forecast', pre, [forecast_tube])):
        if tag == 'fan' and not fan:
            continue
        model2, data2, qadr2, gadr2 = world_model(hose, tubes=tubes)
        r2 = mujoco.Renderer(model2, a.height, a.width)
        SC.render_flags(r2)
        q, g = rest(sol)
        SC.pose(model2, data2, qadr2, gadr2, q, g)
        for view in a.views:
            cam2 = SC.view(model2, view)
            if tubes and tag == 'fan':
                img = SC.opaque_tubes(r2, data2, cam2, opt)    # thin tubes read better solid
            elif tubes:
                img, _ = SC.composite_ghost(r2, data2, cam2, opt, alpha=a.ghost_alpha)
            else:
                r2.update_scene(data2, camera=cam2, scene_option=opt)
                img = r2.render()
            name = 'cycle%02d_%s_%s.png' % (a.cycle, view, tag)
            iio.imwrite(os.path.join(out, name), img)
            written['%s/%s' % (view, tag)] = name
            print('  %-28s %s' % (tag, name))
        r2.close()
        del r2

    manifest = dict(run=run, cycle=a.cycle, phase=phase_name, views=a.views, files=written,
                    chain_worst_mm=float(chain_worst_mm),
                    chain_keyframes=int(sum(len(x['names']) for x in sol.values())),
                    size=[a.width, a.height], ghost='the model\'s forecast for the chosen move',
                    ik_worst_mm={r: float(s['pos_err_mm'].max()) for r, s in sol.items()},
                    rig_arm=meta['rig_arm'])
    open(os.path.join(out, 'manifest.json'), 'w').write(json.dumps(manifest, indent=1) + '\n')
    print('-> %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
