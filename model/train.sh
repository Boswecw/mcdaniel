#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (where this script lives)
cd "$(dirname "$0")"

# Ensure venv exists, then activate it
if [[ ! -d ".venv" ]]; then
  echo "▶ Creating virtual env (.venv)…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Ensure core deps are installed
python - <<'PY' || pip install -q --upgrade scikit-learn pandas numpy scipy joblib
import importlib
for m in ["sklearn","pandas","numpy","scipy","joblib"]:
    importlib.import_module(m)
print("deps-ok")
PY

# Defaults if you run with no args (override by passing flags)
SAMPLES="${SAMPLES:-12000}"
CV="${CV:-3}"
NITER="${NITER:-15}"
SEED="${SEED:-42}"

echo "▶ Training…"
if [[ $# -gt 0 ]]; then
  # You provided flags, we pass them straight through
  python model/train_model.py "$@"
else
  # Use sensible defaults
  python model/train_model.py --samples "$SAMPLES" --cv "$CV" --n-iter "$NITER" --seed "$SEED"
fi
