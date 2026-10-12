#!/bin/bash
# Joint 35+63+78 (BDDL attached), 15000 more mixed-batch steps from joint_35_63_78_noise_step7500 with
# training-only proprio state noise (std 0.05, 23 real dims; user, 2026-10-10). Same LR style:
# warmup 1500 (10%) -> 2e-5 cosine -> 2e-6. 656-row stage table (this branch).
# -> public HF shi2bis/joint_35_63_78/joint_35_63_78_noise_step22500 + READY, then eval.
set -x
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
WT=/workspace/persistent/behavior/wt-attach
DATA=/home/kasm-user/data/2026-challenge-demos
CK=pi_behavior_b1k_fast_100task
EXP=joint356378_noise15000; STEPS=15000; NOISE=0.05
INIT=$WT/outputs/checkpoints/$CK/joint356378_noise3000/2999/params
[ -d $INIT ] || { echo "FATAL: init $INIT missing"; exit 1; }
FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 26 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=26G"; exit 1; }
rm -rf $DATA  # leftover task 0 data (re-downloadable)
python -u - <<'PY' || { echo "FATAL: download failed"; exit 1; }
from huggingface_hub import snapshot_download
keys = ["observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0", "observation.rgb.zed_link_camera_0"]
pats = ["meta/**"] + [p for c in ("035", "063", "078") for p in [f"data/chunk-{c}/**"] + [f"videos/{k}/chunk-{c}/**" for k in keys]]
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset", local_dir="/home/kasm-user/data/2026-challenge-demos", allow_patterns=pats)
print("download done", flush=True)
PY
cd $WT
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$WT/src python -u scripts/train_one_task.py --task-index 35 --ep-start 0 --ep-end 0 \
  --episode-ranges 7000:7200,12600:12800,15600:15800 --weight-loader-params-path $INIT --exp-name $EXP \
  --num-steps $STEPS --corr-noise --batch-size 32 --num-workers 64 --warmup-steps 1500 --peak-lr 2e-5 \
  --decay-steps $STEPS --decay-lr 2e-6 --save-interval 500 --keep-period 100000 --state-noise-std $NOISE
RC=$?
echo "$(date -Is) train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: training failed"; exit 1; }
D=$WT/outputs/checkpoints/$CK/$EXP/$((STEPS-1))
[ -d $D/params ] || { echo "FATAL: no final checkpoint at $D"; exit 1; }
echo "FINAL: $D"
REPO=shi2bis/joint_35_63_78; FOLDER=joint_35_63_78_noise_step22500
python -u scripts/push_ready.py --checkpoint-dir $D --repo $REPO --folder $FOLDER --tasks 35 63 78 --step 22500 \
  --recipe "joint mixed-batch 35+63+78, 15000 more steps from joint_35_63_78_noise_step7500 (22500 total) with proprio state noise std $NOISE (z-score units, 23 real dims, training only), bs 32, corr noise, Ilia stats, LR warmup 1500 -> 2e-5 cosine -> 2e-6, depth videos not loaded" \
  --serve-with "TASK_NUM_STAGES of branch attach-35-63-78 (656 rows). No inference change: noise is training-only."
echo "$(date -Is) push rc=$?"
PYTHONPATH=$WT/src python -u scripts/eval_push.py --checkpoint-dir $D --label joint356378_noise15000 --own-task 35 --extra-tasks 63 78 \
  --known-tasks 1 --log-tag 356378 --hf-repo $REPO --hf-path $FOLDER || echo "$(date -Is) eval FAILED"
echo "$(date -Is) ALL DONE"
