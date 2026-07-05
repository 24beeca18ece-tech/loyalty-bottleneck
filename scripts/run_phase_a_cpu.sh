#!/usr/bin/env bash
# Phase A (CPU, no GPU): reproducible end-to-end entry point.
#
# Creates a local .venv, installs ONLY the CPU-only Phase-A deps (no
# torch/transformers), then runs data generation + the separability audit +
# the unit tests. Uses ./.venv/bin/python throughout so it works regardless of
# whether the venv is activated in your shell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"

echo "[phase-a] repo root: $REPO_ROOT"

# --- 1. venv -----------------------------------------------------------------
if [ ! -x "$PY" ]; then
    echo "[phase-a] creating virtualenv at .venv"
    python3 -m venv "$VENV"
fi

# --- 2. Phase-A deps (CPU-only subset; NO torch/transformers) -----------------
echo "[phase-a] installing CPU-only dependencies (pyyaml, pytest, numpy)"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet pyyaml pytest numpy

# --- 3. data generation + separability audit ---------------------------------
# `python -m src.data_gen` writes outputs/sample_data.jsonl, prints example
# conversations, and runs audit_separability() on a 200/category dataset.
echo "[phase-a] running data generation + separability audit"
"$PY" -m src.data_gen

# --- 4. unit tests -----------------------------------------------------------
echo "[phase-a] running unit tests"
"$PY" -m pytest tests/ -v

# --- done --------------------------------------------------------------------
echo
echo "############################################################"
echo "#                    PHASE A COMPLETE                       #"
echo "############################################################"
