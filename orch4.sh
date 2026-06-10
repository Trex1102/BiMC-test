#!/bin/bash
cd /media/ndag/newVolume/BiMC-test
P=/home/ndag/anaconda3/envs/fscil-env/bin/python
ollama stop qwen3:14b 2>/dev/null; sleep 5
echo "calib start $(date)" >> logs/orch4.log
env PYTHONUNBUFFERED=1 $P dino_calibration_cub.py > logs/calibration.log 2>&1
echo "calib done $(date)" >> logs/orch4.log
env PYTHONUNBUFFERED=1 $P dino_ceiling_robust.py > logs/ceiling_robust.log 2>&1
echo "ceiling done $(date)" >> logs/orch4.log
touch logs/ORCH4_DONE
