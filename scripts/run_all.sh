#!/usr/bin/env bash
# Train every method x seed with 4 parallel workers.  Usage: bash scripts/run_all.sh [physics] [methods...]
set -euo pipefail
PHYSICS=${1:-inline}; shift || true
METHODS=${*:-"nscsd no_shield_discovery no_discovery no_elite no_anchor no_crn grpo_vanilla ppo nscsd_safe no_shield"}
JOBS=${JOBS:-4}
mkdir -p runs/logs
for m in $METHODS; do for s in $(seq 42 51); do
  [ -f "runs/$PHYSICS/$m/seed$s/meta.json" ] || echo "$m $s"
done; done | xargs -P "$JOBS" -L 1 bash -c \
  'python scripts/train.py --method "$0" --seed "$1" --physics '"$PHYSICS"' > runs/logs/'"$PHYSICS"'_$0_$1.log 2>&1'
