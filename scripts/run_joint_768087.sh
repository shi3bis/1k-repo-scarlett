#!/bin/bash
# Joint (mixed-batch) 76 + 80 + 87 training -- the fix for the closed-loop task
# interference seen after the sequential 76 -> 80 -> 87 chain. All three tasks share
# scenes, so with their frames mixed in every batch the same view needs different
# actions per task ID and the model has to use it. Frames are sampled uniformly from
# the union of the three tasks' 200 episodes each (1.84M / 1.74M / 2.04M frames,
# 33/31/36%), i.e. roughly balanced without reweighting.
# From the chain's base ext100_t80t87 (rebuilt if missing), 4500 steps, bs 32,
# corr noise, Ilia stats, LR warmup 450 (10%) -> 2e-5 cosine -> 2e-6.
# Saves every 500 steps on /workspace (persistent). Final kept LOCAL only (user, 2026-10-08;
# private HF quota is full), then evaluated.
set -x
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
R=/workspace/persistent/behavior/behavior-1k-solution; cd $R
EXP=joint768087_4500; STEPS=4500
INIT=/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100_t80t87/params
RANGES=15200:15400,16000:16200,17400:17600
FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 26 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=26G"; exit 1; }
if [ ! -d $INIT ]; then
  JAX_PLATFORMS=cpu python -u scripts/extend_stage_embeddings.py || { echo "FATAL: base rebuild failed"; exit 1; }
fi
python -u - <<'PY' || { echo "FATAL: download failed"; exit 1; }
from huggingface_hub import snapshot_download
keys = ["observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0", "observation.rgb.zed_link_camera_0"]
pats = ["meta/**"] + [p for c in ("076", "080", "087") for p in [f"data/chunk-{c}/**"] + [f"videos/{k}/chunk-{c}/**" for k in keys]]
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset", local_dir="/home/kasm-user/data/2026-challenge-demos", allow_patterns=pats)
print("download done", flush=True)
PY
XLA_PYTHON_CLIENT_PREALLOCATE=false python -u scripts/train_one_task.py --task-index 76 --ep-start 0 --ep-end 0 \
  --episode-ranges $RANGES --weight-loader-params-path $INIT --exp-name $EXP --num-steps $STEPS --corr-noise \
  --batch-size 32 --num-workers 64 --warmup-steps 450 --peak-lr 2e-5 --decay-steps $STEPS --decay-lr 2e-6 \
  --save-interval 500 --keep-period 100000
RC=$?
echo "$(date -Is) train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: training failed"; exit 1; }
D=$R/outputs/checkpoints/pi_behavior_b1k_fast_100task/$EXP/$((STEPS-1))
[ -d $D/params ] || { echo "FATAL: no final checkpoint at $D"; exit 1; }
echo "FINAL: $D"
python -u scripts/eval_push.py --checkpoint-dir $D --label joint768087 --own-task 76 --extra-tasks 80 87 || echo "$(date -Is) eval FAILED"
echo "$(date -Is) ALL DONE"
