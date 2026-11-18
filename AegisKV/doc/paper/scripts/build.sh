#!/usr/bin/env sh
set -eu

AEGISKV_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
AEGISKV_PAPER_DIR=$(CDPATH= cd -- "$AEGISKV_SCRIPT_DIR/.." && pwd)

sh "$AEGISKV_SCRIPT_DIR/check_design_sync.sh"
cd "$AEGISKV_PAPER_DIR"

TEXMFHOME="$AEGISKV_PAPER_DIR/.texmf" \
TEXMFCONFIG="$AEGISKV_PAPER_DIR/.texmf-config" \
TEXMFVAR="$AEGISKV_PAPER_DIR/.texmf-var" \
TEXINPUTS=template/acmart/acmart//: \
BSTINPUTS=template/acmart/acmart//: \
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
