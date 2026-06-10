#!/bin/bash
LOG=/media/ndag/newVolume/BiMC-test/logs/dino_ceiling_res_cub_seed1.log
while ! grep -q "R=518] FSCIL" "$LOG" 2>/dev/null; do
  pgrep -f "dino_ceiling_res_cub[.]py" >/dev/null || exit 0
  sleep 15
done
sleep 3
pkill -f "dino_ceiling_res_cub[.]py"
echo "killed script1 after vitb518 $(date)" >> /media/ndag/newVolume/BiMC-test/logs/orch.log
