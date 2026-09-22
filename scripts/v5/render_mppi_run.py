#!/usr/bin/env python3
"""A real MPPI routing run as a Blender video, side by side with the rig's own film. One command, four stages.

    python scripts/v5/render_mppi_run.py --run ../hose-routing/outputs/mppi_real_v5/run_20260917_150620 \
        [--blender-scene base.blend] [--output mppi_run_visualization.mp4] [--preview] [--frames 890:1544]

Each stage needs a different interpreter, so this file only orchestrates (any python3 runs it):

  1. robot   mild-trackdlo/yam_bimanual/.venv_trace   scripts/v5/mppi_viz_robot.py
             the rig's IK for every executed phase, the session's step schedule, FK body tracks
  2. data    .venv-v5                                  scripts/v5/mppi_viz_data.py
             the timeline: hose, MPPI samples/forecasts, film sync, captions, camera, per output frame, and
             the ghost-arm requests (each shown candidate as the keyframes the rig would have flown)
  3. ghosts  mild-trackdlo/yam_bimanual/.venv_trace   scripts/v5/mppi_viz_robot.py --ghosts
             the rig's IK for every ghost arm (cached on the requests)
  4. render  ~/Projects/blender-robot-render/.venv (bpy 4.5)   render/render_mppi_run.py
             the Blender scene, a Cycles frame per timeline frame, the side-by-side mp4

Everything lands in <run>/blender/ (the "bundle"). Stage 1 is skipped when its outputs exist (--redo-robot
forces it); stage 2 takes seconds; stage 4 resumes with --resume. See scripts/v5/mppi_blender_README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mppi_paths import HOSE_ROUTING_ROOT as ROOT, RENDER_REPO, SCRIPTS
YAM = Path(os.environ.get('YAM_BIMANUAL_DIR', ROOT / 'mild-trackdlo/yam_bimanual')).expanduser().resolve()
PY_ROBOT = YAM / '.venv_trace/bin/python'
PY_DATA = ROOT / '.venv-v5/bin/python'
PY_BLENDER = RENDER_REPO / '.venv/bin/python'


def run(cmd, **kw):
    print('\n$ ' + ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def write_readme(bundle: Path, output: Path) -> None:
    """The bundle's own README, with this run's numbers (like report/ and showcase/ carry)."""
    tl = json.loads((bundle / 'timeline.json').read_text())
    sched = json.loads((bundle / 'robot_schedule.json').read_text())
    sync = '\n'.join('| %s | %.1f | %.1f | %+.2f | %.2f | %.2f |' % (
        s['tag'], s['film_s'], s['schedule_s'], s['offset_s'], s['r'], sched['phases'][s['tag']]['chain_worst_mm'])
        for s in tl['sync'])
    filmed = bool(tl.get('filmed', True))
    panels = ('Left: the run reconstructed\nfrom its own log in Blender, with its legend. Right: the overhead '
              "camera's film of the same moments, untouched." if filmed else
              'One panel: the run rebuilt in\nBlender from its own log. This run is SIMULATED -- it moved grasp '
              'points in Newton and never commanded an\narm -- so there is no camera film beside it, and the arms '
              'are the rig IK put behind the gripper waypoints the\nsim executed.')
    sync_section = ('''## Sync

Every execution frame is one film frame; the Blender panel is evaluated at that frame's capture time. The arm
schedule's start inside each film is not logged, so it is fitted by correlating the film's frame-difference
energy with the schedule's joint speed. The whole-chain column is the rig IK's grasp site against the commanded
keyframe.

| phase | film s | schedule s | fitted offset s | r | whole-chain worst mm |
| --- | --- | --- | --- | --- | --- |
%s''' % sync) if filmed else ('''## Timing

There is no camera to synchronise against: the arm schedule IS the clock, rebuilt exactly as a held session
would have staged these keyframes, and the run clock counts commanded motion only (a simulated run has no wall
clock; the seconds its planner took are shown as pauses). Whole-chain check, the rig IK's grasp site against
the commanded keyframe:

| phase | schedule s | whole-chain worst mm |
| --- | --- | --- |
%s''' % '\n'.join('| %s | %.1f | %.2f |' % (tag, ph['duration_s'], ph['chain_worst_mm'])
                  for tag, ph in sched['phases'].items()))
    num = (lambda v: '%.0f' % v) if tl.get('cost_unit') == 'mm' else (lambda v: '%.2f' % v)
    cyc = '\n'.join('| %d | %s → %s | %s | %.2f – %.2f | %d |' % (
        c['cycle'], num(c['cost_before']), num(c['cost_after']), num(c['forecast']),
        c['color_range'][0], c['color_range'][1], c['drawn_valid']) for c in tl['cycles'])
    segs = '\n'.join('| %.1f – %.1f s | %s |' % (s['t0'], s['t1'], s['label']) for s in tl['segments'])
    colour_word = "the planner's own MPPI cost" if tl['color_by'] == 'planner' else 'the forecast routing cost'
    cost_label = tl.get('cost_label', 'routing cost')
    legacy = bool(tl.get('legacy'))
    checked = int(tl.get('cost_axis_checked') or 0)
    verified = ('The captured-shape contract rebuilt from this run reproduces all %d cycle reports it logged '
                '(mean/max/tip error, peg sides, winding), and every candidate is rebuilt into rig keyframes '
                'that match the phases the rig was given to %.3f mm.\n'
                % (checked, tl.get('rebuild_mm') or 0.)) if legacy else \
               ("The run's own evaluator reproduces every logged cycle cost.\n" if checked else '')
    extra = ('\n**This run is from the earlier campaign** (`scripts/mppi/run_real_mppi.py`): the THIN hose against a\n'
             'captured goal SHAPE with a side/winding contract, so the cost axis is millimetres of mean shape error and\n'
             'the pale blue line is the goal shape. Its record keeps the learned model\'s forecast for the 64 lowest-cost\n'
             'rows of each iteration only, so the ghost arms are drawn from those; a row whose forecast diverged (cost\n'
             'floored at 1e3) is never drawn. It recorded no feasibility screen: reach and grasp failures are penalties\n'
             'inside the cost, not refusals.\n') if legacy else ''
    run_rel = Path(tl['run'])
    # Keep the input absolute: the render checkout and run data live in different repositories.
    text = f"""# MPPI run, rendered in Blender beside the rig's film

`{output.name}`: {tl['n_frames']} frames, {tl['duration_s']:.0f} s at {tl['fps']} fps. {panels}
No titles: the segment table below gives the times to put them at (`--annotate` burns in captions instead).

Rebuild (from the rendering repo): `python scripts/v5/render_mppi_run.py --run {run_rel}`
(method, colour mapping, frames and sync: `scripts/v5/mppi_blender_README.md`).

{verified}{extra}
## Colour

Candidate strokes are coloured by **{colour_word}**, normalised per cycle over its feasible rows:
{tl['colors']['colormap']}. Grey = refused rows. Magenta = the plan MPPI chose, its translucent ghost = the
model's forecast of the hose after it, white = the path the rig's gripper actually drove.

| cycle | {cost_label}{' (mm)' if tl.get('cost_unit') == 'mm' else ''} | forecast | colour range (cost) | feasible strokes drawn |
| --- | --- | --- | --- | --- |
{cyc}

{sync_section}

## Timeline

| video | segment |
| --- | --- |
{segs}
"""
    (bundle / 'README.md').write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--blender-scene', default=None, help='optional base .blend (its world/lights/floor are kept)')
    ap.add_argument('--output', default=None, help='mp4 path (default <run>/blender/mppi_run.mp4)')
    ap.add_argument('--bundle', default=None, help='default <run>/blender')
    ap.add_argument('--preview', action='store_true', help='half-size panels, 16 samples')
    ap.add_argument('--frames', default=None, help='a:b output-frame range to render')
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--annotate', action='store_true', help='captions, run clock and timeline strip around the panels')
    ap.add_argument('--redo-robot', action='store_true')
    ap.add_argument('--color-by', choices=('planner', 'forecast'), default='planner')
    ap.add_argument('--strokes', action='store_true', help='candidates as gripper-path lines, not ghost arms')
    ap.add_argument('--probe-speed', type=float, default=5.)
    ap.add_argument('--cycle-speed', type=float, default=2.)
    a = ap.parse_args(argv)

    run_dir = Path(a.run)
    run_dir = (run_dir if run_dir.is_absolute() else Path.cwd() / run_dir).resolve()
    bundle = Path(a.bundle).resolve() if a.bundle else run_dir / 'blender'
    bundle.mkdir(parents=True, exist_ok=True)
    output = Path(a.output).resolve() if a.output else bundle / ('mppi_run_preview.mp4' if a.preview else 'mppi_run.mp4')
    for py in (PY_ROBOT, PY_DATA, PY_BLENDER):
        if not py.exists():
            sys.exit('missing interpreter %s (see scripts/v5/mppi_blender_README.md, "Environments")' % py)

    # A SIMULATED run sent no phase to a rig, so there are no keyframes to solve: rebuild what each
    # executed action would have commanded first (stage 0), and hand the robot stage those instead.
    phases_dir = bundle / 'phases'
    simulated = not any((run_dir / 'phases').glob('phase_*.json'))
    if simulated and (a.redo_robot or not (phases_dir / 'index.json').exists()):
        run([PY_DATA, SCRIPTS / 'mppi_viz_data.py', '--run', run_dir,
             '--rebuild-phases', phases_dir], cwd=ROOT)
    if a.redo_robot or not (bundle / 'robot_tracks.npz').exists():
        run([PY_ROBOT, SCRIPTS / 'mppi_viz_robot.py', '--run', run_dir, '--out', bundle]
            + (['--phases-dir', phases_dir] if simulated else []),
            cwd=YAM, env=dict(os.environ, MUJOCO_GL='egl'))
    run([PY_DATA, SCRIPTS / 'mppi_viz_data.py', '--run', run_dir, '--bundle', bundle,
         '--color-by', a.color_by, '--probe-speed', a.probe_speed, '--cycle-speed', a.cycle_speed]
        + (['--strokes'] if a.strokes else []), cwd=ROOT)
    if not a.strokes:
        run([PY_ROBOT, SCRIPTS / 'mppi_viz_robot.py', '--run', run_dir, '--out', bundle,
             '--ghosts', bundle / 'ghost_requests.json'],
            cwd=YAM, env=dict(os.environ, MUJOCO_GL='egl'))
    cmd = [PY_BLENDER, RENDER_REPO / 'render/render_mppi_run.py', '--bundle', bundle, '--output', output,
           '--samples', a.samples]
    if a.blender_scene:
        cmd += ['--blender-scene', Path(a.blender_scene).resolve()]
    if a.preview:
        cmd.append('--preview')
    if a.frames:
        cmd += ['--frames', a.frames]
    if a.resume:
        cmd.append('--resume')
    if a.annotate:
        cmd.append('--annotate')
    run(cmd)
    write_readme(bundle, output)
    print('\n-> %s' % output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
