#!/bin/bash
# chain82 sim eval, parameterised copy of run_chain82_8h.sh (OUT, TASKS, INSTANCES, BUDGET_S, MAX_STEPS env vars).
# (2x avg human demo length, task.jsonl), videos. One process per episode. Needs the policy server on :8000.
# Output: /workspace/persistent/eval_outputs_chain82_8h/{metrics,videos,progress.txt}, logs in .../logs/
source /workspace/persistent/env.sh
source /workspace/persistent/miniconda3/etc/profile.d/conda.sh
conda activate behavior
unset DISPLAY                                              # Kasm X (:1) -> xcb errors + segfault
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json  # skip Mesa dzn/lvp ICDs
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES TORCHDYNAMO_DISABLE=1
OUT=${OUT:-/workspace/persistent/eval_outputs_chain82_notricks}
BUDGET_S=${BUDGET_S:-28800}
mkdir -p $OUT/logs
declare -A MAXS=([picking_up_trash]=10535 [re_shelving_library_books]=19056 [dispose_of_glass]=18390 [store_batteries]=15408)
TASKS=(${TASKS:-re_shelving_library_books})
cd /workspace/persistent/repos/place_inside_4_tasks   # repo root: never run from inside BEHAVIOR-1K/ (cwd shadowing)
T0=$(date +%s)
echo "$(date -u) START budget=${BUDGET_S}s" >> $OUT/progress.txt
for k in ${INSTANCES:-1 2 3}; do
  for t in "${TASKS[@]}"; do
    el=$(( $(date +%s) - T0 ))
    if [ $el -ge $BUDGET_S ]; then echo "$(date -u) BUDGET REACHED (${el}s) -- stop" >> $OUT/progress.txt; exit 0; fi
    [ -f $OUT/metrics/${t}_${k}_0.json ] && continue   # resumable
    echo "$(date -u) START round=$k task=$t instance=$k max_steps=${MAX_STEPS:-${MAXS[$t]}}" >> $OUT/progress.txt
    python -u scripts/run_custom_eval.py --task-name $t --instance-ids $k --policy websocket \
      --host 127.0.0.1 --port 8000 --max-steps ${MAX_STEPS:-${MAXS[$t]}} --write-video --log-path $OUT \
      > $OUT/logs/${t}_${k}.log 2>&1
    rc=$?
    res=$(grep -h "RESULT task=" $OUT/logs/${t}_${k}.log | tail -1)
    echo "$(date -u) END round=$k task=$t rc=$rc ${res:-NO_RESULT}" >> $OUT/progress.txt
  done
done
