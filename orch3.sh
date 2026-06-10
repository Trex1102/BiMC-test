#!/bin/bash
cd /media/ndag/newVolume/BiMC-test
P=/home/ndag/anaconda3/envs/fscil-env/bin/python
while [ ! -f logs/ORCH2_DONE ]; do sleep 20; done
echo "orch2 done -> rerank $(date)" >> logs/orch3.log
env PYTHONUNBUFFERED=1 $P dino_concept_rerank_cub.py --partition gt --tag oracle_gt_rerank --resolution 448 --bs 8 > logs/rerank_oracle.log 2>&1
echo "rerank done -> llm_partition $(date)" >> logs/orch3.log
env PYTHONUNBUFFERED=1 $P llm_partition_cub.py > logs/llm_partition.log 2>&1
echo "llm_partition done $(date)" >> logs/orch3.log
touch logs/ORCH3_DONE
