#!/usr/bin/env bash
# Train the experiment grid with $JOBS parallel workers (skips finished runs).
#   bash scripts/run_all.sh [physics]
# Main methods use seeds 42-51; secondary ablations use seeds 42-46.
set -euo pipefail
PHYSICS=${1:-inline}
JOBS=${JOBS:-4}
MAIN=${MAIN:-"nscsd no_shield_discovery ppo grpo_vanilla"}
ABL=${ABL:-"no_discovery no_elite no_anchor no_crn no_shield nscsd_safe"}
mkdir -p runs/logs
{
  for m in $MAIN; do for s in $(seq 42 51); do echo "$m $s"; done; done
  for m in $ABL; do for s in $(seq 42 46); do echo "$m $s"; done; done
} | while read -r m s; do
  [ -f "runs/$PHYSICS/$m/seed$s/meta.json" ] || [ -f "runs/$PHYSICS/$m/seed$s/.running" ] || echo "$m $s"
done | xargs -P "$JOBS" -L 1 bash -c \
  'mkdir -p runs/'"$PHYSICS"'/$0/seed$1 && touch runs/'"$PHYSICS"'/$0/seed$1/.running && python scripts/train.py --method "$0" --seed "$1" --physics '"$PHYSICS"' > runs/logs/'"$PHYSICS"'_$0_$1.log 2>&1; rm -f runs/'"$PHYSICS"'/$0/seed$1/.running'
