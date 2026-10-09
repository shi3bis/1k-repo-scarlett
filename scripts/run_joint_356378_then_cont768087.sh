#!/bin/bash
# Two joint runs back to back (user, 2026-10-09), both with the first joint run's recipe:
# bs 32, corr noise, Ilia stats, LR warmup 450 (10%) -> 2e-5 cosine -> 2e-6, 4500 steps.
#  A) 35 + 63 + 78 (all share the BDDL `attached` goal predicate), mixed batches, from
#     ext100_t80t87_t63t78 (656-row stage table: 63=3, 78=4 stages). Runs in this worktree
#     (branch attach-35-63-78) -> public HF shi2bis/joint_35_63_78 + READY, then eval.
#  B) 76 + 80 + 87 for another 4500 joint steps from the first joint run's final (step 4500),
#     649-row table, main repo (branch task76-resume-2000-lowlr) -> public HF
#     shi2bis/joint_76_80_87/joint_76_80_87_step9000 + READY, then eval.
# Datasets live on the container disk (56GB): A's chunks are deleted before B downloads.
set -x
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
WT=/workspace/persistent/behavior/wt-attach
MAIN=/workspace/persistent/behavior/behavior-1k-solution
DATA=/home/kasm-user/data/2026-challenge-demos
CK=pi_behavior_b1k_fast_100task
STEPS=4500
SCHED="--num-steps $STEPS --corr-noise --batch-size 32 --num-workers 64 --warmup-steps 450 --peak-lr 2e-5 --decay-steps $STEPS --decay-lr 2e-6 --save-interval 500 --keep-period 100000"
SCHED_TXT="bs 32, corr noise, Ilia stats, LR warmup 450 -> 2e-5 cosine -> 2e-6, depth videos not loaded"

check_free() {
  FREE=$(df --output=avail -BG /workspace | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge 26 ] || { echo "FATAL: only ${FREE}G free on /workspace, need >=26G"; exit 1; }
}

download() {  # chunk numbers, rgb only
  python -u - "$@" <<'PY' || { echo "FATAL: download failed"; exit 1; }
import sys
from huggingface_hub import snapshot_download
keys = ["observation.rgb.left_realsense_link_camera_0", "observation.rgb.right_realsense_link_camera_0", "observation.rgb.zed_link_camera_0"]
pats = ["meta/**"] + [p for c in sys.argv[1:] for p in [f"data/chunk-{c}/**"] + [f"videos/{k}/chunk-{c}/**" for k in keys]]
snapshot_download("behavior-1k/2026-challenge-demos", repo_type="dataset", local_dir="/home/kasm-user/data/2026-challenge-demos", allow_patterns=pats)
print("download done", sys.argv[1:], flush=True)
PY
}

# ---------------- A) 35 + 63 + 78 ----------------
echo "$(date -Is) A: joint 35+63+78"
check_free
cd $WT
EXP=joint356378_4500
INIT=/workspace/persistent/behavior/checkpoints/behavior_50t_checkpoint_ext100_t80t87_t63t78/params
[ -d $INIT ] || { echo "FATAL: missing $INIT (scripts/extend_stage_embeddings.py)"; exit 1; }
download 035 063 078
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$WT/src python -u scripts/train_one_task.py --task-index 35 --ep-start 0 --ep-end 0 \
  --episode-ranges 7000:7200,12600:12800,15600:15800 --weight-loader-params-path $INIT --exp-name $EXP $SCHED
RC=$?
echo "$(date -Is) A train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: A training failed"; exit 1; }
DA=$WT/outputs/checkpoints/$CK/$EXP/$((STEPS-1))
[ -d $DA/params ] || { echo "FATAL: no final checkpoint at $DA"; exit 1; }
echo "A FINAL: $DA"
REPO_A=shi2bis/joint_35_63_78; FOLDER_A=joint_35_63_78_step4500
python -u scripts/push_ready.py --checkpoint-dir $DA --repo $REPO_A --folder $FOLDER_A --tasks 35 63 78 --step 4500 \
  --recipe "joint mixed-batch 35+63+78 (BDDL attached; episodes 7000-7199, 12600-12799, 15600-15799, frames sampled uniformly), 4500 steps from the ext100_t80t87_t63t78 base, $SCHED_TXT" \
  --serve-with "TASK_NUM_STAGES of branch attach-35-63-78 (656 rows: 649-row table + 63=3, 78=4)"
RC=$?
echo "$(date -Is) A push rc=$RC"
PYTHONPATH=$WT/src python -u scripts/eval_push.py --checkpoint-dir $DA --label joint356378 --own-task 35 --extra-tasks 63 78 \
  --known-tasks 1 --log-tag 356378 --hf-repo $REPO_A --hf-path $FOLDER_A || echo "$(date -Is) A eval FAILED"
[ $RC -eq 0 ] || { echo "FATAL: A push failed; not starting B (A's data and checkpoint kept)"; exit 1; }
rm -rf $DATA
echo "$(date -Is) A done; dataset dir removed"

# ---------------- B) 76 + 80 + 87, another 4500 ----------------
echo "$(date -Is) B: joint 76+80+87 continuation"
check_free
cd $MAIN
EXP=joint768087_cont4500
INIT=$MAIN/outputs/checkpoints/$CK/joint768087_4500/4499/params
[ -d $INIT ] || { echo "FATAL: missing $INIT"; exit 1; }
download 076 080 087
XLA_PYTHON_CLIENT_PREALLOCATE=false python -u scripts/train_one_task.py --task-index 76 --ep-start 0 --ep-end 0 \
  --episode-ranges 15200:15400,16000:16200,17400:17600 --weight-loader-params-path $INIT --exp-name $EXP $SCHED
RC=$?
echo "$(date -Is) B train exited rc=$RC"
[ $RC -eq 0 ] || { echo "FATAL: B training failed"; exit 1; }
DB=$MAIN/outputs/checkpoints/$CK/$EXP/$((STEPS-1))
[ -d $DB/params ] || { echo "FATAL: no final checkpoint at $DB"; exit 1; }
echo "B FINAL: $DB"
REPO_B=shi2bis/joint_76_80_87; FOLDER_B=joint_76_80_87_step9000
python -u $WT/scripts/push_ready.py --checkpoint-dir $DB --repo $REPO_B --folder $FOLDER_B --tasks 76 80 87 --step 9000 \
  --recipe "joint mixed-batch 76+80+87 (episodes 15200-15399, 16000-16199, 17400-17599), 4500 more steps from joint_76_80_87_step4500 (9000 total; LR re-warmed), $SCHED_TXT" \
  --serve-with "TASK_NUM_STAGES of branch task76-resume-2000-lowlr (649 rows, ext100_t80t87)"
echo "$(date -Is) B push rc=$?"
python -u scripts/eval_push.py --checkpoint-dir $DB --label joint768087_cont4500 --own-task 76 --extra-tasks 80 87 \
  --hf-repo $REPO_B --hf-path $FOLDER_B || echo "$(date -Is) B eval FAILED"
echo "$(date -Is) ALL DONE"
