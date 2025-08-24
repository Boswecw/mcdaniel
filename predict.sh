#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d ".venv" ]]; then
  echo "❌ .venv not found. Run ./train.sh once (it will create it), or create it via:"
  echo "   python3 -m venv .venv && source .venv/bin/activate && pip install scikit-learn pandas numpy scipy joblib"
  exit 1
fi
source .venv/bin/activate
if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <distance> <size:0|1> <minute_of_day> <dow:0-6> <month:1-12> [extra args]"
  echo "Example: $0 120 1 600 3 9"
  exit 1
fi
DIST="$1"; SIZE="$2"; MIN="$3"; DOW="$4"; MON="$5"
shift 5
python model/predict.py --distance "$DIST" --size "$SIZE" --minute "$MIN" --dow "$DOW" --month "$MON" "$@"
