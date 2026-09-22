

## Quickstart (GaP only)

```bash
# 1. Replay one trial (run from graph-as-policy so libero is on PYTHONPATH)
cd ~/graph-as-policy
uv run python ~/blender-robot-render/render/replay.py \
    --trial-dir outputs/libero_crate_washing/eval/task_00/trial_15_rc1_reward1.000_pass \
    --out ~/Projects/blender-robot-render/outputs/trial_15

# 2. Build + render (uses the local bpy env)
cd ~/blender-robot-render
uv run python scripts/render_trial.py \
    --traj outputs/trial_15/traj.npz \
    --mjcf outputs/trial_15/scene_combined.xml \
    --camera agentview \
    --out outputs/trial_15/render.mp4
```


## Peg-climb policy renders (xArm7 + LEAP hand)



### One render (best of a few seeds)

```bash
cd ~/vibereact          # repo root with the POLICY package

# 1. Roll out the policy, pick the best episode, dump scene.json / traj.npz.
PYTHONPATH=$PWD MUJOCO_GL=egl conda run -n lerobot --no-capture-output python \
    thirdparty/blender-robot-render/render/replay_peg_climb.py \
    --policy output/policies/vibeact_paper_peg_climb_dense_seed3/best_success.zip \
    --out   thirdparty/blender-robot-render/outputs/peg_climb_seed3 \
    --seeds 1 0 4 3 --randomize

# 2. Assemble the .blend (light-blue peg, textured robot).
conda run -n blender-render python \
    thirdparty/blender-robot-render/render/build_scene_pegclimb.py \
    --replay-dir   thirdparty/blender-robot-render/outputs/peg_climb_seed3 \
    --output-blend thirdparty/blender-robot-render/outputs/peg_climb_seed3/scene.blend

# 3. Render frames + mp4 (enables every OPTIX GPU).
conda run -n blender-render python \
    thirdparty/blender-robot-render/render/render_pegclimb.py \
    --blend   thirdparty/blender-robot-render/outputs/peg_climb_seed3/scene.blend \
    --out-dir thirdparty/blender-robot-render/outputs/peg_climb_seed3/frames \
    --video   thirdparty/blender-robot-render/outputs/peg_climb_seed3/peg_climb.mp4 \
    --samples 160 --fps 30
```

The peg colour is the `PEG_LIGHT_BLUE` constant in `build_scene_pegclimb.py`.

### Batch: N trials end-to-end

```bash
# Renders N=10 trials (one rollout per seed) -> per-trial mp4. Safe to run in tmux.
bash thirdparty/blender-robot-render/scripts/render_peg_climb_trials.sh 10
```



## MPPI hose-routing run, side by side with the rig's film

The complete rendering pipeline is in `scripts/v5/` and `render/render_mppi_run.py`:
YAM robot tracks, the hose/MPPI timeline, candidate ghost arms, and a Blender panel
beside the overhead camera film. Planner/IK libraries, meshes and recorded runs
come from the sibling `hose-routing` checkout (override with `HOSE_ROUTING_ROOT`).

```bash
cd ~/Projects/blender-robot-render
python scripts/v5/render_mppi_run.py \
    --run ../hose-routing/outputs/mppi_real_v5/run_20260917_150620 \
    --output outputs/mppi_run_visualization.mp4
# Add --preview --frames 890:1544 to render just cycle 1 at reduced quality.
```

To render an already prepared bundle, use the Blender environment directly:

```bash
.venv/bin/python render/render_mppi_run.py \
    --bundle ../hose-routing/outputs/mppi_real_v5/run_20260917_150620/blender \
    --out-dir outputs/mppi_run/frames \
    --output outputs/mppi_run_visualization.mp4
```

The export stages use hose-routing's `.venv-v5` and the YAM stack's `.venv_trace`;
the rendering stage uses this repo's `.venv` (bpy 4.5). Prepared bundles reference
external film frames and meshes, so those assets must remain available.
See [the pipeline documentation](scripts/v5/mppi_blender_README.md) for flags,
input formats, colour mapping, coordinate frames and synchronisation.
