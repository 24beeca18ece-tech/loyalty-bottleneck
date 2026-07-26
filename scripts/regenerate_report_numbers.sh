#!/usr/bin/env bash
# Regenerate every result JSON cited in REPORT.md, from organism_v3, in one go.
#
# These JSONs are gitignored and are NOT in a fresh clone; this script rebuilds
# them so REPORT.md's numbers become mechanically verifiable again. It needs a
# CUDA GPU (the real Qwen-2.5-1.5B organism) -- it will NOT run on CPU.
#
# Usage (tomorrow, on the RTX 4060):
#     ./scripts/regenerate_report_numbers.sh
# Optional overrides via env vars:
#     ADAPTER=outputs/organism_v3  MODEL=Qwen/Qwen2.5-1.5B-Instruct  DEVICE=cuda
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ADAPTER="${ADAPTER:-outputs/organism_v3}"   # the best-selectivity checkpoint (see REPORT.md §2.1)
DEVICE="${DEVICE:-cuda}"

echo "[regen] repo root : $REPO_ROOT"
echo "[regen] model     : $MODEL"
echo "[regen] adapter   : $ADAPTER"
echo "[regen] device    : $DEVICE"

# --- guard 0: venv python exists ---------------------------------------------
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Create the venv and install deps first:" >&2
    echo "  python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

# --- guard 1: CUDA GPU required ----------------------------------------------
if ! "$PY" - <<'PYEOF'
import sys
try:
    import torch
except ImportError:
    print("ERROR: torch is not installed in .venv.", file=sys.stderr)
    print("  ./.venv/bin/python -m pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: no CUDA GPU available.", file=sys.stderr)
    print("  This script regenerates the REAL Qwen-2.5-1.5B organism results and", file=sys.stderr)
    print("  requires a CUDA GPU (e.g. the RTX 4060). It will not run on CPU.", file=sys.stderr)
    print("  (For a CPU smoke test of the pipeline shape, use: "
          "./.venv/bin/python scripts/run_probe_demo.py --tiny)", file=sys.stderr)
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
print(f"[gpu] {props.name}  ({props.total_memory / 1e9:.1f} GB VRAM)")
PYEOF
then
    exit 1
fi

# --- guard 2: adapter must exist ---------------------------------------------
if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: adapter directory '$ADAPTER' not found." >&2
    echo "  It is gitignored and not in a fresh clone. Train it first:" >&2
    echo "  ./.venv/bin/python -m src.train_organism --output-dir $ADAPTER" >&2
    echo "  (or point ADAPTER=... at an existing checkpoint)." >&2
    exit 1
fi

# --- stage 1: affordance sweep -> outputs/affordance_results.json -------------
echo
echo "############ STAGE 1/4: affordance sweep (Finding 1, REPORT.md §3) ############"
"$PY" scripts/run_affordance_eval.py \
    --model "$MODEL" --adapter "$ADAPTER" --device "$DEVICE" \
    --out outputs/affordance_results.json \
    --plot-out outputs/figures/fig1_affordance_headline.png

# --- stage 2: selectivity diagnostic -> outputs/organism_v3/diagnostic.json ---
echo
echo "############ STAGE 2/4: selectivity diagnostic (REPORT.md §2.1 table) ############"
"$PY" scripts/diagnose_organism.py \
    --adapter-dir "$ADAPTER" --base-model "$MODEL" \
    --out "$ADAPTER/diagnostic.json"

# --- stage 3: principal-specificity -> outputs/specificity_{mean,last}.json ---
# Finding 2 (§4) reports three pooling strategies. run_probe_demo exposes the two
# CLI pooling modes (mean, last); the third (mention-span) is a diagnostic that
# is not a --pooling option here -- regenerate it separately if needed.
echo
echo "############ STAGE 3/4: principal-specificity, mean + last pooling (Finding 2, §4) ############"
"$PY" scripts/run_probe_demo.py \
    --model "$MODEL" --adapter "$ADAPTER" --device "$DEVICE" \
    --pooling mean --out outputs/specificity_mean.json
"$PY" scripts/run_probe_demo.py \
    --model "$MODEL" --adapter "$ADAPTER" --device "$DEVICE" \
    --pooling last --out outputs/specificity_last.json

# --- stage 4: causal steering -> outputs/steering_results.json ----------------
echo
echo "############ STAGE 4/4: causal steering (Finding 3, REPORT.md §5) ############"
"$PY" scripts/run_steering_eval.py \
    --model "$MODEL" --adapter "$ADAPTER" --device "$DEVICE" \
    --out outputs/steering_results.json

# --- done --------------------------------------------------------------------
echo
echo "############################################################"
echo "#           REPORT NUMBERS REGENERATED                     #"
echo "############################################################"
echo "Wrote:"
echo "  outputs/affordance_results.json        (Finding 1 / §3 table)"
echo "  $ADAPTER/diagnostic.json               (§2.1 selectivity table)"
echo "  outputs/specificity_mean.json          (Finding 2 / §4, mean pooling)"
echo "  outputs/specificity_last.json          (Finding 2 / §4, last-token pooling)"
echo "  outputs/steering_results.json          (Finding 3 / §5)"
echo
echo "Note: the mention-span pooling row in §4 is a separate diagnostic and is"
echo "not produced by --pooling (only mean/last are CLI options)."
echo "Next: reconcile these against outputs/NUMBER_VERIFICATION_CHECKLIST.md."
