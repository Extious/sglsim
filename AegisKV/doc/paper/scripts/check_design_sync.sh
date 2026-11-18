#!/usr/bin/env sh
set -eu

AEGISKV_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
AEGISKV_DESIGN="$AEGISKV_SCRIPT_DIR/../../design/current_design.md"
AEGISKV_SECTION="$AEGISKV_SCRIPT_DIR/../sections/03-system-design-v2.tex"
AEGISKV_ALG_SU="$AEGISKV_SCRIPT_DIR/../algorithms/slack-update.tex"
AEGISKV_ALG_SL="$AEGISKV_SCRIPT_DIR/../algorithms/slack-landlord.tex"
AEGISKV_ALG_GA="$AEGISKV_SCRIPT_DIR/../algorithms/guarded-aimd.tex"

AEGISKV_ACTUAL=$(sha256sum "$AEGISKV_DESIGN" | awk '{print $1}')

for AEGISKV_RENDERED in "$AEGISKV_SECTION" "$AEGISKV_ALG_SU" "$AEGISKV_ALG_SL" "$AEGISKV_ALG_GA"; do
  AEGISKV_EXPECTED=$(sed -n 's/^% \(Source\|Design\)-SHA256: //p' "$AEGISKV_RENDERED")
  if [ -z "$AEGISKV_EXPECTED" ]; then
    echo "design-sync error: fingerprint is missing from $AEGISKV_RENDERED" >&2
    exit 1
  fi
  if [ "$AEGISKV_EXPECTED" != "$AEGISKV_ACTUAL" ]; then
    echo "design-sync error: $AEGISKV_RENDERED is stale relative to current_design.md." >&2
    echo "Synchronize the section and all three algorithm files, then use:" >&2
    echo "$AEGISKV_ACTUAL" >&2
    exit 1
  fi
done

echo "design-sync ok: $AEGISKV_ACTUAL"
