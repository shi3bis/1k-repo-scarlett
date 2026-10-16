#!/bin/bash
# Task 0, +10000 steps from the HF state-noise final (task0_turning_on_radio_statenoise3000,
# 10000 total from ext100) -> 20000 total (user, 2026-10-16). Same recipe as noise3000:
# bs 32, corr noise, state noise 0.05, warmup 1000 (10%) -> 2e-5 cosine -> 2e-6.
# -> public HF shi2bis/task0_turning_on_radio_statenoise/task0_turning_on_radio_statenoise_step20000 + READY, then eval.
set -x
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
B=/workspace/persistent/behavior; WT0=$B/wt-task0
P=$WT0/outputs/checkpoints/pi_behavior_b1k_fast_100task
REPO=shi2bis/task0_turning_on_radio_statenoise; SRC=task0_turning_on_radio_statenoise3000
HFDIR=$WT0/outputs/hf_init
INIT=$HFDIR/$SRC/params
EXP=task0_noise_cont10000; STEPS=10000; NOISE=0.05; WARM=1000
DATA=/home/kasm-user/data/2026-challenge-demos
FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 40 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=40G"; exit 1; }
python -u - <<PY || { echo "FATAL: init download failed"; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("$REPO", local_dir="$HFDIR", allow_patterns=["$SRC/params/**", "$SRC/READY"])
print("init download done", flush=True)
PY
[ -f $HFDIR/$SRC/READY ] && [ -d $INIT ] || { echo "FATAL: init $INIT missing/not READY"; exit 1; }
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
  --warmup-steps $WARM --peak-lr 2e-5 --decay-steps $STEPS --decay-lr 2e-6 --save-interval 500 --keep-period 100000 \
  --state-noise-std $NOISE
RC=$?
echo "$(date -Is) train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: training failed"; exit 1; }
D=$P/$EXP/$((STEPS-1))
[ -d $D/params ] || { echo "FATAL: no final checkpoint at $D"; exit 1; }
echo "FINAL: $D"
FOLDER=task0_turning_on_radio_statenoise_step20000
python -u $B/wt-attach/scripts/push_ready.py --checkpoint-dir $D --repo $REPO --folder $FOLDER --tasks 0 --step 20000 \
  --recipe "10000 steps from $SRC (HF; 20000 total from ext100), proprio state noise std $NOISE (z-score units, 23 real dims, training only), bs 32, corr noise, Ilia stats, LR warmup $WARM -> 2e-5 cosine -> 2e-6, depth videos not loaded" \
  --serve-with "ext100 TASK_NUM_STAGES (627 rows; branch task0-radio-2000). No inference change: noise is training-only."
echo "$(date -Is) push rc=$?"
PYTHONPATH=$WT0/src python -u scripts/eval_push.py --checkpoint-dir $D --label task0_noise_cont10000 --own-task 0 \
  --hf-repo $REPO --hf-path $FOLDER || echo "$(date -Is) eval FAILED"
echo "$(date -Is) ALL DONE"
