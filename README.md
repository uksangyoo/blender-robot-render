

## Quickstart

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

