# MPPI run → Blender, side by side with the rig's film

Run the commands below from `blender-robot-render`. These scripts were copied from
`hose-routing/scripts/v5`; rendering helpers live here, while the planner, rig IK,
robot meshes, recordings and the two export environments remain in `hose-routing`.
By default it is a sibling checkout; set `HOSE_ROUTING_ROOT=/path/to/hose-routing`
for another location. `YAM_BIMANUAL_DIR` can override the rig stack directory and
`BLENDER_ROBOT_RENDER` can override the Blender checkout/environment.
The `.venv-v5` and `mild-trackdlo/...` paths below are relative to `HOSE_ROUTING_ROOT`.
A prepared bundle can be rendered directly without running the export stages;
it still references the original film frames and robot mesh files.

```bash
python scripts/v5/render_mppi_run.py --run ../hose-routing/outputs/mppi_real_v5/run_20260917_150620
# -> outputs/mppi_real_v5/run_20260917_150620/blender/mppi_run.mp4  (+ README.md with this run's numbers)
```

Left panel: the run reconstructed in Blender from its own log — arms, hose, pegs, goal sides, and for every
MPPI cycle the sampled candidates as **ghost arms** flying their moves, coloured by cost, the model's forecasts,
and the chosen plan as a magenta ghost arm before the real arm executes it. Right panel: the overhead camera's film
of the same moments, untouched. The only text is the cost colour bar ("MPPI cost", low → high), shown while the
coloured ghost arms are on screen; titles are meant to be added in editing (segment times in `<run>/blender/README.md`). Style after
`~/Projects/blender-robot-render/chair upright.mp4` (that clip is only a style reference; nothing in the render
repo generates it).

| flag | use |
| --- | --- |
| `--preview` | half-size panels, 16 samples (full quality: ~0.5 s/frame, ~1.5 s with ghost arms; ~45 min for this run) |
| `--frames 890:1544` | one output-frame range — this one is cycle 1: planning, execution, result |
| `--resume` | keep frames already rendered |
| `--blender-scene base.blend` | start from an existing scene: its world, lights and `floor` are kept |
| `--color-by forecast` | colour candidates by the re-scored forecast instead of the planner's cost |
| `--probe-speed 5 --cycle-speed 2` | film playback speed in probe and cycle segments |
| `--strokes` | draw candidates as gripper-path lines (the first version) instead of ghost arms |
| `--peg-labels` | draw p0/p1/p2 above the pegs (render stage flag; off by default: the video carries no text but the cost colour bar) |
| `--annotate` | add captions (cycle, stage, counts), the film's run clock / pause badge and a bottom timeline strip; by default the video is only the render with its legend beside the untouched film |

Captions only: `~/Projects/blender-robot-render/.venv/bin/python render/render_mppi_run.py --bundle <run>/blender --compose-only`
re-encodes from the rendered frames in seconds.

## Which runs this renders

Three shapes of run go in, and the pipeline works out which from the directory itself. Everything below the
first row is unchanged between them: same colouring, same ghost arms, same camera.

| run | what changes |
| --- | --- |
| **`run_real_v5.py` on the rig** (`outputs/mppi_real_v5/run_*`, `outputs/v5_real_campaign/**`) | the reference case: two panels, peg-side routing cost, the film's clock. |
| **`run_real_mppi.py` on the rig** (`outputs/mppi_real/run_*`, the earlier campaign) | read through `showcase_data_legacy.py`: the thin 22 mm hose against a captured goal SHAPE with a side/winding contract over four 20 mm pegs. The goal shape is drawn (pale blue), the cost axis is millimetres of mean shape error, and candidates come from the 64 rows per iteration whose forecast the recorder kept. It has no `observations/`, no `events.jsonl`, no `run.log` and no `plan_file`: the state the planner held is the film's own phase header, the settled state is the next `trajectory['states']` row, and phases are paired to actions rather than to positions (one probe of run_20260911_222321 wrote no phase file, which shifts every later one). |
| **a simulated run** (`--rig sim`: `outputs/v5_goal_ladder/**`, `outputs/v5_g2_real_sim/*`, …) | no camera, so ONE panel and no sync to fit: the arm schedule is the clock and the run clock counts commanded motion only. It also sent no phase to a rig, so stage 0 rebuilds what each executed action would have commanded (`--rebuild-phases`) and the arms are the rig's IK behind the gripper waypoints the sim executed -- a reconstruction, which the video's own README and its top-right tag say. |

Each of the three is checked against its own record before anything is drawn: the v5 evaluator reproduces every
logged cycle cost, the earlier campaign's contract reproduces every logged `shape_after` (mean/max/tip error,
peg sides, winding), and in both campaigns rebuilding an executed action gives back the keyframes the rig was
given (0.000 mm on run_20260911_212456, _222321 and _223820).

## Pipeline

Three interpreters, four stages; `scripts/v5/render_mppi_run.py` in this repository runs them in order and writes `<run>/blender/`.

| stage | interpreter | script | writes |
| --- | --- | --- | --- |
| phases (only for a run with no `phases/`) | `.venv-v5` | `scripts/v5/mppi_viz_data.py --rebuild-phases` | `<bundle>/phases/<tag>.json` + `index.json` |
| robot | `mild-trackdlo/yam_bimanual/.venv_trace` | `scripts/v5/mppi_viz_robot.py` | `robot_scene.json`, `robot_tracks.npz`, `robot_schedule.json` |
| data | `.venv-v5` | `scripts/v5/mppi_viz_data.py` | `timeline.json`, `timeline.npz`, `ghost_requests.json` |
| ghosts | `mild-trackdlo/yam_bimanual/.venv_trace` | `scripts/v5/mppi_viz_robot.py --ghosts` | `ghost_tracks.npz` (cached on the requests; ~3 min) |
| render | `~/Projects/blender-robot-render/.venv` (bpy 4.5.14) | `render/render_mppi_run.py` | `render/f*.png`, the mp4 |

The render venv is the render repo's own pyproject (Python 3.11, `bpy>=4.5,<4.6`); on this machine it was made with
`~/miniforge3/envs/yam311/bin/python -m venv .venv && .venv/bin/pip install "bpy>=4.5,<4.6" "numpy<2" Pillow pyyaml "imageio[ffmpeg]"`.
With a Blender binary instead: `blender -b -P render/render_mppi_run.py -- --bundle ...` (arguments after `--`).

Reused, not re-derived: `showcase_data` (observations, samples, `hand_paths`, the run's `PegSideEvaluator`,
`check()` which asserts the evaluator reproduces every logged cycle cost), `render_showcase_robot.solve_phase` +
`showcase_scene.build` (the rig's pyroki IK and the composed two-arm MuJoCo world),
`execute_phase_yam.split_stages/stage_of` (keyframe staging), and in the render repo `build_scene` (mesh import,
PBR materials, Cycles/OptiX setup), `build_scene_pegclimb.build_bodies_and_geoms` and
`replay_peg_climb.build_scene_graph` (now takes an optional `mesh_files` map).

## Run data it expects

| file | used for |
| --- | --- |
| `metrics.json` | probes/cycles in order, `plan_file`, costs, `planning.predicted_states`, `planning_pegs` |
| `config/<task>.json`, `config/rig.yaml` | the pegs and goal, and the rig the ghost/rebuild planner uses |
| `observations/<tag>.npz` | `pre_centers`/`post_centers` (the 28-node hose the planner used), `planned_mind_action` |
| `samples/cycle_NNN_iter_I.npz` | every candidate: `actions` (1024, 2, 20), `costs`, `weights`, `valid`, `refused`, `pred` (1024, 2, 28, 3) |
| `phases/phase_NNN.json` | the keyframes the rig executed, `meta.rig_arm`, `meta.sync`, base transforms |
| `film/<tag>/frames.jsonl` + `f*.jpg` | the RGB panel, and each frame's capture time `t` (absent in a sim run: one panel) |
| `events.jsonl` | run start, plan times, `planning_seconds`, the z-fit report |
| `run.log` | the held session's `--seg-time/--first-time/--close-time/--home-time` |
| `calibration.json` | overhead K and `T_world_overhead` (whole-chain check only) |

For the earlier campaign, the same information comes from different files: `film/<tag>/frames.jsonl`'s phase
header (the state and action), `trajectory.npz` (`states`, `goal`, `physical_goal_line`), `film/calibration.json`
(the overhead pose, `cable_radius_m`, the pegs) and `samples/*.npz` (`costs`, `actions` (K, H, 21),
`best_rollouts`, which are the rollouts of `argsort(costs)[:64]`). A row at cost ≥ 1e3 is one whose forecast
diverged (`planning/goal.py:262`) and is never drawn; that campaign recorded no feasibility screen, so its reach
and grasp failures are penalties inside the cost rather than refusals.

Tags: `probe_00k`, `probe_00k_retryN` for a retried probe, `cycle_00n`.

## Coordinate frames

- **One world frame for everything**: the rig world (metres, z up, table top at z = 0.75, the arm bases at
  x ≈ 0.25, y = ±0.30, pegs at x ≈ 0.60). Hose nodes, pegs, landmarks, sample paths and forecasts are all already
  in it. Blender is also metres and z up, so there is no axis change; MuJoCo quaternions are wxyz, as Blender's are.
- **Arms**: the phase JSON's keyframes are world positions; `solve_phase` inverts `T_world_leftbase` /
  `T_world_rightbase` (from the phase's `meta`) for IK, and the composed MuJoCo world attaches each arm at its
  base transform, so FK body poses come out in world. `meta.rig_arm` maps the planner's roles
  (`minus_y`/`plus_y`) to physical arms and is read, never assumed. Checked per phase: `grasp_site` vs the
  commanded keyframe, worst 0.48 mm.
- **Jaws**: the i2rt slide runs qpos 0 = shut → 0.0475 = open, so `qpos = grip × 0.0475` (measured on the pad
  meshes; the tip body origins move the other way and mislead). The jaws follow the gripper readback the runner
  logged after every keyframe (`metrics[...]['grip']`): ~0.997 open, ~0.6 shut on the 70 mm hose, which stops them.
- **Candidate paths**: one action phase is `[arc0, arc1, 2 hands × 3 waypoints × (dx, dy, dz)/0.12]`. The grasp
  node is `round(arc × 27)` on the state the planner held, and waypoints are cumulative deltas × 0.12 m from it
  (`showcase_data.hand_paths`, the release's `decode_action`).
- **Overhead camera** (whole-chain check only): OpenCV (+z forward, +y down) → Blender camera (−z forward, +y up)
  is `R_blender = R_cv · diag(1, −1, −1)`; the film is the calibration's 1280×720 at 0.75, so K is scaled to 960×540.
  `--view overhead --still N` renders through it and writes render | film | 50/50. Base poses, IK, hose frames and
  the sync offset all have to agree for the overlay to line up — it did at the grasp, mid-carry and after release.

## Ghost arms: each candidate as the rig would have flown it

For each MPPI iteration, 6 candidates spread evenly over the feasible rows' cost ranking (the best included) become
ghost arms. A candidate's first move goes through **the runner's own path from action to rig keyframes**
(`mppi_viz_data.RigPlanner`): `run_real_v5.overshoot_phase`, the protocol's carry rule (`lifted` rows flown at
carry height), `release_clearance`, then `act.executor.phase_to_motion_plan` with the run's rig config. The
result is a `phases/*.json`-form plan, solved with the same `solve_phase` IK as the executed phases. Checked on
this run: the four executed cycles' logged actions rebuild their `phases/*.json` keyframes to 0.00 mm, and the
geometric arm assignment below reproduces all four executed assignments.

- **Arm assignment**: the grasp with the lower y goes to the −y arm. The rows are never rebuilt as
  `PhaseAction(role, a[0], a[1])` from the sorted 20-number vector: that crossed the arms on the rig.
- **What moves**: a ghost is the arm whose motion keyframes travel more than 10 mm. It shows `close` → lift →
  carry → place, joint-linear between keyframes as `move_joints` drives them, with the jaws at the shut readback
  (0.6). 12 of this run's 76 ghost plans move both arms, so both are drawn. Approach and retreat are left out.
- **Refusals**: a plan whose motion keyframes the IK rejects is dropped (none were here, as every drawn row had
  already passed the planner's reach screen). The planner's refused rows are not drawn as arms.
- **Timing**: ghosts fade in worst-first at their grasp pose and sweep their move over about 1 s, and the next
  iteration replaces them. The last iteration's arms stay, dimmed, while their forecast hoses appear in the same
  colour. Then the magenta ghost arm flies the chosen plan (the logged phase itself), before the real arm executes.

## How MPPI cost maps to colour

Per cycle, over the feasible rows of all three iterations:
`u = (cost − min) / (max − min)`, colour **green = lowest cost (good) → amber → red = highest (bad)**
(saturated stops in `mppi_viz_data.cost_colour`, not RdYlGn, whose pale middle vanishes against the white hose).
Ghost opacity falls with `u` (0.40 → 0.28). `cost` is the planner's own objective (`samples['costs']`, the number its weights
`exp(−(c − min)/T)` were computed from). It is staged and ties at 0 for many rows (cycle 1: ~90% of feasible rows
≤ 0.03); tied rows got equal weight, so they share a colour. `--color-by forecast` instead re-scores each row's
forecast with the run's routing objective (continuous; 0.58–1.0 correlated with the planner's cost here).

- **ghost arms**, green → red: the candidates above, coloured by cost.
- **thin coloured hoses**: the learned model's forecast after each ghost's move (`pred[:, 0]`), in the same colour.
- **magenta arm**: the plan that ran. This is the importance-weighted mean of the lowest-cost rows, re-screened and
  verified, so it is not one of the 1024 rows. The **magenta hose** is its forecast, `planning.predicted_states[0]`.
- Only each candidate's **first** move is shown, the move that would execute now. The candidate is still coloured
  by its whole two-step cost.
- With `--strokes` (the first version), candidates are drawn as gripper-path lines instead, with up to 40 grey
  refused rows per iteration and a white line for the path the rig drove.

## How the two panels are synchronised

- **Execution segments**: every output frame is one film frame, played at `--probe-speed` / `--cycle-speed` times
  real time. The Blender panel is evaluated at that frame's capture time `t`.
- **Arm schedule**: rebuilt exactly as the held session staged it (`yam_session.run_phase` + `drive_lockstep`:
  the first move 2.4 s, `close` 2.0 s, other steps 1.2 s, synced arms in parallel, homing right then left, 2.0 s
  each), and it matches the `[ARM] name over X s` lines in `run.log`.
- **Offset inside the film**: not logged, so it is fitted per phase by cross-correlating frame-difference energy
  with the schedule's joint speed. For this run it came out at +0.34 … +0.58 s, r = 0.69–0.80, over all 11 phases.
- **Planning, intro, z-fit and result segments** pause the real clock and the RGB panel holds a frame. With
  `--annotate` it is dimmed and shows "PAUSED · planning took 6.5 s", plus the run-time clock.

## What is and is not in the log

- **Hose**: the planner's observed pre and post states. It moves only while the carry runs, blended pre → post
  and paced by the gripper's path length. That blend is interpolation, not dynamics. The film's per-frame tracker
  is not used mid-move: it agrees with the settled observations to ~7 mm at both ends, but lags under the carrying
  arm and its ends flail during the retreat.
- **Peg sides** change only when an observation arrives.
- **Retreat pose**: the arms retreat to joint zero, which is assumed. `yam_session.starts_ref` is never written,
  but the film's first frame shows that folded pose and the overhead overlay agrees.
- **Cycle 1 pause**: there was a ~10 min pause between planning and executing. The video cuts it; the clock shows it.
