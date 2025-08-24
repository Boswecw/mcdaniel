#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d ".venv" ]]; then
  echo "▶ Creating virtual env (.venv)…"
  python3 -m venv .venv
fi
source .venv/bin/activate
python - <<'PY' || pip install -q --upgrade scikit-learn pandas numpy scipy joblib
import importlib
for m in ["sklearn","pandas","numpy","scipy","joblib"]:
    importlib.import_module(m)
print("deps-ok")
PY
SAMPLES="${SAMPLES:-12000}"
CV="${CV:-3}"
NITER="${NITER:-15}"
SEED="${SEED:-42}"
echo "▶ Training…"
if [[ $# -gt 0 ]]; then
  python model/train_model.py "$@"
else
  python model/train_model.py --samples "$SAMPLES" --cv "$CV" --n-iter "$NITER" --seed "$SEED"
fi
