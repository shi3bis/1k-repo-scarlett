#!/bin/bash
# Joint 57+60+69+93 (all have the BDDL `covered` predicate in their goal), 6000 mixed-batch steps
# from ext100_t80t87_t63t78_cov4 (673-row stage table: 57=5, 60=4, 69=3, 93=5), with training-only
# proprio state noise std 0.05. Same LR style as the previous runs: warmup 600 (10%) -> 2e-5
# cosine -> 2e-6. bs 32, corr noise, Ilia stats. (user, 2026-10-10)
# -> public HF shi2bis/joint_57_60_69_93/joint_57_60_69_93_step6000 + READY, then eval.
# Waits for the joint 35+63+78 noise run (same GPU) to exit first.
set -x
# PID of run_joint_356378_noise3000.sh (pgrep -f would also match shells quoting its name)
while kill -0 3473435 2>/dev/null; do sleep 60; done
echo "$(date -Is) previous run exited; starting"
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
WT=/workspace/persistent/behavior/wt-covered
DATA=/home/kasm-user/data/2026-challenge-demos
CK=pi_behavior_b1k_fast_100task
EXP=joint_cov4_6000; STEPS=6000; NOISE=0.05
INIT=/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100_t80t87_t63t78_cov4/params
[ -d $INIT ] || { echo "FATAL: init $INIT missing (scripts/extend_stage_embeddings.py)"; exit 1; }
FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 26 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=26G"; exit 1; }
rm -rf $DATA  # previous run's 35/63/78 data (re-downloadable)
python -u - <<'PY' || { echo "FATAL: download failed"; exit 1; }
from huggingface_hub import snapshot_download
keys = ["observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0", "observation.rgb.zed_link_camera_0"]
pats = ["meta/**"] + [p for c in ("057", "060", "069", "093") for p in [f"data/chunk-{c}/**"] + [f"videos/{k}/chunk-{c}/**" for k in keys]]
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset", local_dir="/home/kasm-user/data/2026-challenge-demos", allow_patterns=pats)
print("download done", flush=True)
PY
cd $WT
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$WT/src python -u scripts/train_one_task.py --task-index 57 --ep-start 0 --ep-end 0 \
  --episode-ranges 11400:11600,12000:12200,13800:14000,18600:18800 --weight-loader-params-path $INIT --exp-name $EXP \
  --num-steps $STEPS --corr-noise --batch-size 32 --num-workers 64 --warmup-steps 600 --peak-lr 2e-5 \
  --decay-steps $STEPS --decay-lr 2e-6 --save-interval 500 --keep-period 100000 --state-noise-std $NOISE
RC=$?
echo "$(date -Is) train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: training failed"; exit 1; }
D=$WT/outputs/checkpoints/$CK/$EXP/$((STEPS-1))
[ -d $D/params ] || { echo "FATAL: no final checkpoint at $D"; exit 1; }
echo "FINAL: $D"
REPO=shi2bis/joint_57_60_69_93; FOLDER=joint_57_60_69_93_step6000
python -u scripts/push_ready.py --checkpoint-dir $D --repo $REPO --folder $FOLDER --tasks 57 60 69 93 --step 6000 \
  --recipe "joint mixed-batch 57+60+69+93 (BDDL covered; episodes 11400-11599, 12000-12199, 13800-13999, 18600-18799, frames sampled uniformly), 6000 steps from the ext100_t80t87_t63t78_cov4 base, proprio state noise std $NOISE (training only), bs 32, corr noise, Ilia stats, LR warmup 600 -> 2e-5 cosine -> 2e-6, depth videos not loaded" \
  --serve-with "TASK_NUM_STAGES of branch covered-57-60-69-93 (673 rows: 656-row table + 57=5, 60=4, 69=3, 93=5). No inference change: noise is training-only."
echo "$(date -Is) push rc=$?"
PYTHONPATH=$WT/src python -u scripts/eval_push.py --checkpoint-dir $D --label joint_cov4_6000 --own-task 57 --extra-tasks 60 69 93 \
  --known-tasks 1 35 --log-tag cov4 --hf-repo $REPO --hf-path $FOLDER || echo "$(date -Is) eval FAILED"
echo "$(date -Is) ALL DONE"
