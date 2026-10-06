#!/usr/bin/env bash
# Build the supplementary-material archive for the IMBANG letter.
#   bash scripts/make_supplementary.sh
#   -> dist/IMBANG_supplementary_material.pdf  (supplementary document, built from paper/nl/supp)
#   -> dist/IMBANG_supplementary.zip           (code, data, checkpoints and the same PDF)
# Packs code, tests, the per-seed results behind every table and figure, the trained
# 10-seed checkpoints of the methods reported in the paper, and the rendered figures.
# Sentinel's own checkpoints are not redistributed; see supplementary/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
NAME=IMBANG_supplementary
TMP=$(mktemp -d)
STAGE=$TMP/$NAME
METHODS="span span_nosafe span_cgrpo span_cgrpo_safe span_entropy span_abscost nscsd ppo grpo_vanilla maxmc_only"
SCRIPTS="train.py run_all.sh evaluate.py span_report.py cliff_probe.py intensity_sweep.py human_in_loop.py
         mechanism_probe.py diagnostics.py icc_assets.py nl_assets.py safe_member_check.py supp_assets.py"

mkdir -p "$STAGE"/{scripts,results,runs/inline,reference_figures}
cp supplementary/README.md NOTICE requirements.txt "$STAGE"/
cp -r nscsd tests "$STAGE"/
for s in $SCRIPTS; do cp scripts/$s "$STAGE"/scripts/; done

# results: merged 10-seed files only (pilot runs and per-worker shards are left out)
mkdir -p "$STAGE"/results/full
cp results/full/*.csv results/full/*.md "$STAGE"/results/full/
mkdir -p "$STAGE"/results/human && cp results/human/*.csv results/human/*.md "$STAGE"/results/human/
cp results/{cliff_probe_by_seed.csv,cliff_probe_final_by_seed.csv,CLIFF_PROBE.md,CLIFF_PROBE_FINAL.md} "$STAGE"/results/
cp results/{intensity_sweep_by_seed.csv,safe_member_check_by_seed.csv,mechanism_probe.csv,MECHANISM.md,DIAGNOSTICS.md} "$STAGE"/results/
cp results/{reproduction.csv,headroom.csv} "$STAGE"/results/

# trained checkpoints, laid out as the scripts expect (runs/inline/<method>/seed<k>/)
for m in $METHODS; do cp -r runs/inline/$m "$STAGE"/runs/inline/; done

cp paper/nl/figures/{mechanism,intensity,dynamics,tradeoff}.pdf "$STAGE"/reference_figures/
cp paper/nl/supp/imbang_supplementary.pdf "$STAGE"/IMBANG_supplementary_material.pdf

find "$STAGE" \( -name __pycache__ -o -name '*.pyc' -o -name .running -o -name .pytest_cache \) -prune -exec rm -rf {} +
mkdir -p dist
rm -f dist/$NAME.zip
cp paper/nl/supp/imbang_supplementary.pdf dist/IMBANG_supplementary_material.pdf
(cd "$TMP" && zip -qr -X "$ROOT/dist/$NAME.zip" $NAME)
rm -rf "$TMP"
echo "wrote dist/$NAME.zip ($(du -h dist/$NAME.zip | cut -f1))"
