#!/bin/bash
# Mid-run push (user, 2026-10-11): run step 10000 of joint356378_noise15000 (= total step 17500) -> public HF
# shi2bis/joint_35_63_78/joint_35_63_78_noise_step17500 + READY, for the eval pod. The run keeps only its latest
# save (every 500 steps), so copy the step dir to the overlay as soon as it is finalized, then push from the copy.
source /workspace/persistent/behavior/env.sh >/dev/null 2>&1
conda activate behavior
WT=/workspace/persistent/behavior/wt-attach
SRC=$WT/outputs/checkpoints/pi_behavior_b1k_fast_100task/joint356378_noise15000/10000
CP=/home/kasm-user/ckpt_joint356378_step17500
until [ -f $SRC/_CHECKPOINT_METADATA ]; do sleep 20; done
sleep 30
echo "$(date -Is) found $SRC, copying"
cp -a $SRC $CP
RC=$?
[ $RC -eq 0 ] || { echo "FATAL: copy failed rc=$RC"; exit 1; }
du -sh $CP
# Intermediate saves carry only norm_stats.json: train_one_task.py bundles fast_tokenizer/ into the FINAL
# step dir only, and push_ready.py refuses READY without it. Add the one the run trains with (as done by
# hand on 2026-10-11 after the first attempt failed that check).
cp -a $WT/outputs/assets_ilia/pi_behavior_b1k_fast_100task/behavior-1k/2026-challenge-demos/fast_tokenizer \
  $CP/assets/behavior-1k/2026-challenge-demos/ || { echo "FATAL: tokenizer copy failed"; exit 1; }
cd $WT
python -u scripts/push_ready.py --checkpoint-dir $CP --repo shi2bis/joint_35_63_78 --folder joint_35_63_78_noise_step17500 \
  --tasks 35 63 78 --step 17500 \
  --recipe "joint mixed-batch 35+63+78, intermediate save at step 10000 of a 15000-step continuation from joint_35_63_78_noise_step7500 (17500 total) with proprio state noise std 0.05 (z-score units, 23 real dims, training only), bs 32, corr noise, Ilia stats, LR warmup 1500 -> 2e-5 cosine (over 15000) -> 2e-6, depth videos not loaded" \
  --serve-with "TASK_NUM_STAGES of branch attach-35-63-78 (656 rows). No inference change: noise is training-only."
RC=$?
echo "$(date -Is) push rc=$RC"
[ $RC -eq 0 ] && rm -rf $CP && echo "$(date -Is) local copy removed"
