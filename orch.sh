#!/bin/bash
cd /media/ndag/newVolume/BiMC-test
P=/home/ndag/anaconda3/envs/fscil-env/bin/python
# wait for Script 1 (resolution sweep) to finish
while pgrep -f "dino_ceiling_res_cub.py" >/dev/null; do sleep 30; done
echo "script1 done $(date)" >> logs/orch.log
env PYTHONUNBUFFERED=1 $P dino_ceiling_fixed_cub.py > logs/ceiling_fixed.log 2>&1
echo "ceiling_fixed done $(date)" >> logs/orch.log
env PYTHONUNBUFFERED=1 $P dino_artifact_gatedfusion_cub.py > logs/artifact_gated.log 2>&1
echo "artifact_gated done $(date)" >> logs/orch.log
touch logs/ORCH_DONE
