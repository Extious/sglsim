#!/usr/bin/env sh
set -eu

AEGISKV_ASSET_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
AEGISKV_PAPER_DIR=$(CDPATH= cd -- "$AEGISKV_ASSET_DIR/.." && pwd)
AEGISKV_REPO_ROOT=$(CDPATH= cd -- "$AEGISKV_PAPER_DIR/../.." && pwd)
AEGISKV_BUILD_DIR="$AEGISKV_PAPER_DIR/tmp/pdfs/evaluation-assets"
AEGISKV_OUTPUT_DIR="$AEGISKV_REPO_ROOT/output/pdf"

mkdir -p "$AEGISKV_BUILD_DIR" "$AEGISKV_OUTPUT_DIR"
cd "$AEGISKV_ASSET_DIR"

TEXMFHOME="$AEGISKV_PAPER_DIR/.texmf" \
TEXMFCONFIG="$AEGISKV_PAPER_DIR/.texmf-config" \
TEXMFVAR="$AEGISKV_PAPER_DIR/.texmf-var" \
TEXINPUTS="$AEGISKV_PAPER_DIR/template/acmart/acmart//:" \
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="$AEGISKV_BUILD_DIR" figure-proof.tex

cp "$AEGISKV_BUILD_DIR/figure-proof.pdf" \
  "$AEGISKV_OUTPUT_DIR/aegiskv-sglsim-figure-insertion-proof.pdf"
