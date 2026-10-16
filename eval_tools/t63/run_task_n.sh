#!/bin/bash
# N instances of $TASK, one sim process per instance (crash-isolated), resumable.
source /workspace/persistent/env.sh
source $OV/miniconda3/etc/profile.d/conda.sh && conda activate behavior
unset DISPLAY; export OMNIGIBSON_HEADLESS=1 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
OUT=${OUT:?}; INSTANCES=${INSTANCES:-"1 2 3 4 5 6 7 8 9 10"}; MAX=${MAX_STEPS:?}; TASK=${TASK:?}
mkdir -p $OUT/logs; cd $REPO
for i in $INSTANCES; do
  if [ -f $OUT/metrics/${TASK}_${i}_0.json ]; then echo "skip $i (done)" >> $OUT/progress.txt; continue; fi
  echo "$(date -u +%FT%T) START inst $i" >> $OUT/progress.txt
  python -u scripts/run_custom_eval.py --task-name ${TASK} --instance-ids $i --policy websocket \
    --host 127.0.0.1 --port ${PORT:-8000} --max-steps $MAX --write-video --log-path $OUT > $OUT/logs/inst$i.log 2>&1
  rc=$?
  q=$(python -c "import json;d=json.load(open('$OUT/metrics/${TASK}_${i}_0.json'));print(d['q_score']['final'],d['success'],d['steps'])" 2>/dev/null || echo NO_METRICS)
  echo "$(date -u +%FT%T) END inst $i rc=$rc q/success/steps=$q" >> $OUT/progress.txt
done
echo "$(date -u +%FT%T) ALL DONE" >> $OUT/progress.txt
