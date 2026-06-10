#!/bin/bash
cd /media/ndag/newVolume/BiMC-test
P=/home/ndag/anaconda3/envs/fscil-env/bin/python
while pgrep -f "dino_concept_dirs_cub.py" >/dev/null; do sleep 15; done
echo "concept_oracle done $(date)" >> logs/orch2.log
env PYTHONUNBUFFERED=1 $P dino_ceiling_fixed_cub.py > logs/ceiling_fixed2.log 2>&1
echo "ceiling done $(date)" >> logs/orch2.log
env PYTHONUNBUFFERED=1 $P dino_artifact_gatedfusion_cub.py > logs/artifact_gated2.log 2>&1
echo "artifact done $(date)" >> logs/orch2.log
touch logs/ORCH2_DONE
