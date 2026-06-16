#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_pipeline.sh — Full Taxonomy Pipeline
# ============================================================
# Usage:
#   ./run_pipeline.sh                  # full pipeline (skip bootstrap)
#   ./run_pipeline.sh --bootstrap 500  # bootstrap taxonomy from top-N panels first
#   ./run_pipeline.sh --enrich-only    # only run Ollama enrichment + visualize
#   ./run_pipeline.sh --pipeline-only  # only run ETL pipeline + visualize
#   ./run_pipeline.sh --rebuild-gold   # force rebuild gold from silver
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOOTSTRAP_N=""
MODE="full"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERR]${NC} $1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bootstrap)    BOOTSTRAP_N="$2"; shift 2 ;;
    --top-n)        BOOTSTRAP_N="$2"; shift 2 ;;  # alias
    --enrich-only)  MODE="enrich-only"; shift ;;
    --pipeline-only) MODE="pipeline-only"; shift ;;
    --rebuild-gold) MODE="rebuild-gold"; shift ;;
    -h|--help)      head -22 "$0"; exit 0 ;;
    *)              log_err "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================================"
echo "  TAXONOMY PIPELINE"
echo "  Mode: $MODE"
[[ -n "$BOOTSTRAP_N" ]] && echo "  Bootstrap top-N: $BOOTSTRAP_N"
echo "============================================================"
echo ""

run_pipeline() {
  log_step "Running ETL pipeline (src/main.py)..."
  python src/main.py
  log_ok "Pipeline complete"
}

run_bootstrap() {
  log_step "Bootstrapping taxonomy from gold (top $BOOTSTRAP_N panels)..."
  python scripts/bootstrap_taxonomy.py --top-n "$BOOTSTRAP_N"
  log_ok "Taxonomy bootstrapped"
}

run_enrich() {
  log_step "Running Ollama classification (src/taxonomy.py --enrich)..."
  python src/taxonomy.py --enrich
  log_ok "Enrichment complete"
}

run_visualize() {
  log_step "Visualizing gold data (src/visualize.py)..."
  python src/visualize.py
  log_ok "Visualization complete"
}

run_rebuild_gold() {
  log_step "Rebuilding gold from silver..."
  python -c "
import pandas as pd
from src.main import read_silver, build_gold, write_gold
df = read_silver()
print(f'Read {len(df):,} silver rows')
gold = build_gold(df)
write_gold(gold)
print(f'Gold written: {len(gold):,} rows')
"
  log_ok "Gold rebuilt"
}

case "$MODE" in
  "pipeline-only")
    run_pipeline
    run_visualize
    ;;

  "enrich-only")
    run_enrich
    run_visualize
    ;;

  "rebuild-gold")
    run_rebuild_gold
    run_visualize
    ;;

  "full")
    # Step 1: Run ETL pipeline
    run_pipeline

    # Step 2: Bootstrap taxonomy if requested
    if [[ -n "$BOOTSTRAP_N" ]]; then
      run_bootstrap
      # Re-run pipeline to pick up new taxonomy.json
      log_step "Re-running pipeline to apply new taxonomy..."
      run_pipeline
    fi

    # Step 3: Ollama enrichment
    run_enrich

    # Step 4: Visualize
    run_visualize
    ;;
esac

echo ""
echo "============================================================"
log_ok "All done!"
echo "============================================================"
