#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_sampling.sh
bash scripts/run_optimization.sh
