#!/usr/bin/env bash
set -euo pipefail

if ! command -v nbstripout >/dev/null 2>&1; then
  echo "nbstripout is not installed. Install via: pip install nbstripout" >&2
  exit 1
fi

find notebooks -name "*.ipynb" -print0 | xargs -0 -I{} nbstripout {}
echo "Stripped outputs from notebooks."
