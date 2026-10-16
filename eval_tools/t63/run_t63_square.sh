#!/bin/bash
# Task-63 client rules (RULES.md best public setup) + SQUARE-UP (pitch/yaw squaring + keeper) + micro grid around the dot roll, public 301-310.
export B1K_SPLIT=public_test
export B1K_GRASP_DESCEND=1
export B1K_WALL_HOLD=1.70 B1K_WALL_EXACT=1
export B1K_WALL_YAWLOCK=1 B1K_WALL_YAWLOCK_VX=${B1K_WALL_YAWLOCK_VX:-0.5}
export B1K_WALL_DOT=1 B1K_WALL_DOT_OFF=177 B1K_WALL_DOT_RIGHT=1
export B1K_WALL_SQUARE=1 B1K_WALL_DOT_OFFS=${DOT_OFFS:-"0,-4,4,-8,8,-12,12"} B1K_WALL_DOT_LAT=${DOT_LAT:-"-0.015,0.015"}
export B1K_WALL_TWIST=1 B1K_WALL_FAST=1 B1K_WALL_PRESS=0.10 B1K_WALL_SEAT=8 B1K_WALL_NOLOCK=1 B1K_WALL_ZERO_FIRST=1
export B1K_WALL_REACH=0.78 B1K_WALL_CREEP_PRESS=1 B1K_WALL_PULL_BASE=1
export TASK=installing_smoke_detectors MAX_STEPS=${MAX_STEPS:-5137} INSTANCES=${INSTANCES:-"301 302 303 304 305 306 307 308 309 310"}
exec /workspace/persistent/diag/run_task_n.sh
