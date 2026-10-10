#!/bin/bash
# Task 0, 3000 more steps from the continuation final (cont500to5000/4499, i.e. 7000 total from
# ext100) with training-only Gaussian noise on the normalized proprio state (std 0.05, real 23 dims;
# user, 2026-10-10). Same LR style as the previous runs: warmup 300 (10%) -> 2e-5 cosine -> 2e-6.
# -> public HF shi2bis/task0_turning_on_radio_statenoise + READY, then eval (own 0, known 1/35).
set -x
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
B=/workspace/persistent/behavior; WT0=$B/wt-task0
P=$WT0/outputs/checkpoints/pi_behavior_b1k_fast_100task
INIT=$P/task0_radio_cont500to5000/4499/params
EXP=task0_noise3000; STEPS=3000; NOISE=0.05
DATA=/home/kasm-user/data/2026-challenge-demos
[ -d $INIT ] || { echo "FATAL: init $INIT missing"; exit 1; }
FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 26 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=26G"; exit 1; }
# Leftover 76/80/87 data from the last joint run (re-downloadable); task 0 needs the room.
rm -rf $DATA
python -u - <<'PY' || { echo "FATAL: download failed"; exit 1; }
from huggingface_hub import snapshot_download
keys = ["observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0", "observation.rgb.zed_link_camera_0"]
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset", local_dir="/home/kasm-user/data/2026-challenge-demos",
                  allow_patterns=["meta/**", "data/chunk-000/**"] + [f"videos/{k}/chunk-000/**" for k in keys])
print("download done", flush=True)
PY
cd $WT0
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$WT0/src python -u scripts/train_one_task.py --task-index 0 --ep-start 0 --ep-end 200 \
  --weight-loader-params-path $INIT --exp-name $EXP --num-steps $STEPS --corr-noise --batch-size 32 --num-workers 64 \
  --warmup-steps 300 --peak-lr 2e-5 --decay-steps $STEPS --decay-lr 2e-6 --save-interval 500 --keep-period 100000 \
  --state-noise-std $NOISE
RC=$?
echo "$(date -Is) train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: training failed"; exit 1; }
D=$P/$EXP/$((STEPS-1))
[ -d $D/params ] || { echo "FATAL: no final checkpoint at $D"; exit 1; }
echo "FINAL: $D"
REPO=shi2bis/task0_turning_on_radio_statenoise; FOLDER=task0_turning_on_radio_statenoise3000
python -u $B/wt-attach/scripts/push_ready.py --checkpoint-dir $D --repo $REPO --folder $FOLDER --tasks 0 --step 3000 \
  --recipe "3000 steps from task0_turning_on_radio_cont5000 (10000 total from ext100), proprio state noise std $NOISE (z-score units, 23 real dims, training only), bs 32, corr noise, Ilia stats, LR warmup 300 -> 2e-5 cosine -> 2e-6, depth videos not loaded" \
  --serve-with "ext100 TASK_NUM_STAGES (627 rows; branch task0-radio-2000). No inference change: noise is training-only."
echo "$(date -Is) push rc=$?"
PYTHONPATH=$WT0/src python -u scripts/eval_push.py --checkpoint-dir $D --label task0_noise3000 --own-task 0 \
  --hf-repo $REPO --hf-path $FOLDER || echo "$(date -Is) eval FAILED"
echo "$(date -Is) ALL DONE"
