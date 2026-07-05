#!/usr/bin/env bash
# Phase A (CPU, no GPU): scaffold sanity + data generation + CPU unit tests.
# Uses a tiny dummy model — no GPU or large downloads required.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[phase-a] repo root: $REPO_ROOT"

# TODO: generate benign organism data + matched controls (CPU).
#   python -m src.data_gen ...

# TODO: run CPU unit tests with the tiny dummy model.
#   python -m pytest tests/ -q

echo "[phase-a] scaffold only — fill in TODOs in a later step."
