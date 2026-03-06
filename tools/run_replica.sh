#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/replica/replica.py"

for idx in $(seq 6 8); do
  echo "=== scene_idx=${idx} ==="
  python sem_gauss.py "$CONFIG" --scene_idx "$idx"
  sleep 10
done