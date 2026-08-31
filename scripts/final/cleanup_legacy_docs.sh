#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE="$ROOT/docs/archive/legacy_bundle_docs"
mkdir -p "$ARCHIVE"
legacy=(
  APPLY.md APPLY_FRONTEND.md CHANGES_V2.md FRONTEND_OVERVIEW.md INTEGRATION_GAPS.md
  VALIDATION.md VALIDATION_FRONTEND.md VALIDATION_V2.md VALIDATION_STOCHASTIC.md
  APPLY_STOCHASTIC.md STOCHASTIC_TRANSFER_GUIDE.md START_HERE_RETRAINING.md TRAINING_GUIDE.md
)
for name in "${legacy[@]}"; do
  if [[ -f "$ROOT/$name" ]]; then
    mv "$ROOT/$name" "$ARCHIVE/$name"
    echo "archived $name"
  fi
done
printf '\nCanonical docs are now under docs/final/ and FINAL_RELEASE.md.\n'
