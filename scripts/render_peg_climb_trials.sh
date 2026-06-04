#!/usr/bin/env bash
# Render N peg-climb policy trials to photoreal Blender mp4s.
# Each trial = one deterministic rollout (seed=i) -> scene.json/traj.npz
#            -> assembled .blend -> Cycles/OPTIX animation -> mp4.
#
# Usage:  render_peg_climb_trials.sh [N_TRIALS]      (default 10)
# Env:    SAMPLES=160  POLICY=<path/to/best_success.zip>
#
# Designed to run detached in tmux; continues past a failed trial.

set -u

REPO=/home/uyoo/vibereact
LEROBOT_PY=/home/uyoo/miniconda3/envs/lerobot/bin/python
BLENDER_PY=/home/uyoo/miniconda3/envs/blender-render/bin/python
RENDER=$REPO/thirdparty/blender-robot-render/render
POLICY=${POLICY:-$REPO/output/policies/vibeact_paper_peg_climb_dense_seed3/best_success.zip}
OUTBASE=$REPO/thirdparty/blender-robot-render/outputs/peg_climb_trials
PUB=$REPO/output/peg_climb_trials
N=${1:-10}
SAMPLES=${SAMPLES:-160}

mkdir -p "$OUTBASE" "$PUB"
cd "$REPO"

echo "=== peg-climb trial render: N=$N samples=$SAMPLES policy=$POLICY ==="
echo "=== started $(date) ==="

ok=0
for i in $(seq 0 $((N - 1))); do
  idx=$(printf "%02d" "$i")
  TDIR=$OUTBASE/trial_$idx
  MP4=$TDIR/peg_climb_trial_$idx.mp4
  echo
  echo "######## TRIAL $i  ($idx) -> $TDIR  $(date +%H:%M:%S) ########"

  echo "[$idx] rollout (lerobot)..."
  PYTHONPATH=$REPO MUJOCO_GL=egl "$LEROBOT_PY" \
      "$RENDER/replay_peg_climb.py" \
      --policy "$POLICY" --out "$TDIR" --seeds "$i" --randomize \
    || { echo "[$idx] ROLLOUT FAILED"; continue; }

  echo "[$idx] build scene (blender-render)..."
  "$BLENDER_PY" "$RENDER/build_scene_pegclimb.py" \
      --replay-dir "$TDIR" --output-blend "$TDIR/scene.blend" \
    || { echo "[$idx] BUILD FAILED"; continue; }

  echo "[$idx] render animation (blender-render)..."
  "$BLENDER_PY" "$RENDER/render_pegclimb.py" \
      --blend "$TDIR/scene.blend" --out-dir "$TDIR/frames" \
      --video "$MP4" --samples "$SAMPLES" --fps 30 \
    || { echo "[$idx] RENDER FAILED"; continue; }

  cp "$MP4" "$PUB/" 2>/dev/null || true
  # quick stats from the trajectory
  "$LEROBOT_PY" - "$TDIR/traj.npz" <<'PY' 2>/dev/null || true
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
print(f"[stats] seed={int(d['seed'])} climb={float(d['climb']):.4f}m "
      f"success={bool(d['success'])} frames={d['body_pos'].shape[0]}")
PY
  ok=$((ok + 1))
  echo "[$idx] DONE  ($ok/$N ok)"
done

echo
echo "=== ALL TRIALS DONE: $ok/$N succeeded.  $(date) ==="
echo "=== mp4s in: $PUB ==="
ls -la "$PUB"/*.mp4 2>/dev/null
