# Taxonomy Pipeline — Moroccan Public Procurement Classification

A 5-level taxonomy pipeline that classifies raw procurement panel names from Moroccan public procurement data (marchés publics) into a structured hierarchy, with LLM-based validation.

## Architecture

```
Raw Panel + Designation
    │
    ▼
normalize_panel()  ──► strip verbs, noise, units
    │
    ▼
canonicalize_panel() ──► regex patterns from taxonomy.json (5 levels)
    │
    ▼
build_gold() ──► recomputes all 5 levels from latest taxonomy.json
    │
    ▼
taxonomy.py --enrich ──► Ollama classification + adversarial validation
    │
    ▼
visualize.py ──► 5-level coverage stats, tree, canonical consolidation
```

## Taxonomy Levels

| Level | Description | Example |
|-------|-------------|---------|
| **L1** | Broad domain | Informatique, Fournitures de Bureau, Restauration |
| **L2** | Medium category | Impression, Papeterie, Repas |
| **L3** | Specific category | Consommables Impression, Écriture, Déjeuner |
| **L4** | Subcategory | Toner, Imprimante, Portable |
| **L5** | Very specific | HP 85A, 256GB, Standard |

## Quick Start

```bash
# Full pipeline: bootstrap taxonomy → run ETL → enrich → visualize
./run_pipeline.sh --bootstrap 500

# Just run ETL + visualize (skip bootstrap/enrich)
./run_pipeline.sh --pipeline-only

# Only Ollama enrichment (after gold exists)
./run_pipeline.sh --enrich-only

# Force rebuild gold from silver
./run_pipeline.sh --rebuild-gold
```

## Key Commands

| Command | Description |
|---------|-------------|
| `python src/main.py` | Run ETL pipeline (produces gold parquet) |
| `python src/taxonomy.py --enrich` | Ollama classify unmatched panels |
| `python src/taxonomy.py --bootstrap` | Generate taxonomy.json from gold top-N |
| `python src/taxonomy.py --report` | Print taxonomy tree summary |
| `python src/visualize.py` | Show 5-level taxonomy coverage & stats |
| `python scripts/bootstrap_taxonomy.py --top-n 500` | Bootstrap taxonomy from top 500 panels |

## Configuration

Key environment variables (or edit `src/config/settings.py`):

```bash
OLLAMA_MODEL="qwen2.5:7b-instruct"           # Main classifier model
OLLAMA_VALIDATION_MODEL="qwen2.5:3b-instruct" # Validation model
VALIDATION_CONFIDENCE_THRESHOLD=0.85          # Min confidence for validation
PANEL_MIN_FREQ_FOR_CLASSIFY=5                 # Min frequency to trigger Ollama
OLLAMA_BASE_URL="http://localhost:11434"
```

## Data Flow

1. **Raw CSV** (`data/raw/Article.csv.gz`) → **Bronze** (chunked parquet)
2. **Bronze** → **Silver** (normalized, validated, deduplicated)
3. **Silver** → **Gold** (AI-ready, filtered, sorted, 5-level taxonomy applied)
4. **Gold** → **Enrichment** (Ollama classifies unmatched panels with validation)
5. **Visualization** → Terminal dashboard with 5-level stats

## Files

```
src/
├── main.py              # ETL pipeline (normalize, canonicalize, build gold)
├── classifier.py        # LLM classifier + adversarial validation
├── taxonomy.py          # Enrichment, bootstrap, reporting
├── visualize.py         # Terminal dashboard (5-level stats)
├── config/
│   ├── settings.py      # Paths, Ollama config, thresholds
│   ├── taxonomy.json    # Current 2-level patterns (legacy)
│   └── taxonomy_5level.json  # 5-level taxonomy patterns
scripts/
└── bootstrap_taxonomy.py  # One-time bootstrap wrapper
run_pipeline.sh            # Orchestrator script
```

## Validation Layer

The pipeline includes an **adversarial validation** step that challenges the classifier:

1. **Path Coherence** — L1→L2→L3→L4→L5 must form valid descending path
2. **Panel-Canonical Alignment** — Canonical must match panel semantics
3. **Specificity Calibration** — Specific panels need L4/L5, vague panels stop at L2/L3
4. **Domain Plausibility** — Cross-domain checks (fuel≠Informatique, medical≠Bureau)
5. **Confidence Consistency** — Flag over/under-confidence

If validation fails → re-classify from scratch with corrected output.

## Example Classification

```
Input:
  Panel: "achat de toner pour imprimante"
  Designation: "cartouche HP LaserJet 85A noir"

Output:
  Canonical: "Toner / Cartouche HP 85A"
  L1: "Informatique"
  L2: "Impression"
  L3: "Consommables Impression"
  L4: "Toner"
  L5: "HP 85A"
  Confidence: 0.95
  Match rationale: "Panel indicates toner purchase; designation specifies HP 85A cartridge"
```

## Output Files

| File | Description |
|------|-------------|
| `data/gold/Article_ai_ready.parquet` | Main output (1.7M rows, 5-level taxonomy) |
| `data/taxonomy/taxonomy_map.parquet` | Panel → 5-level lookup table |
| `data/taxonomy/classifier_cache.json` | Ollama cache (persistent across runs) |
| `data/audit/report_*.json` | Quality reports |

## Requirements

- Python 3.10+
- Ollama running locally (`ollama serve`)
- Models: `qwen2.5:7b-instruct` (classifier), `qwen2.5:3b-instruct` (validation)
- Python packages: `pandas`, `pyarrow`, `rich`, `requests`, `tqdm`

Install deps:
```bash
pip install pandas pyarrow rich requests tqdm python-dotenv
```

## Customization

- **Add patterns**: Edit `src/config/taxonomy_5level.json` → add regex entries with 5 levels
- **Adjust thresholds**: Modify `VALIDATION_CONFIDENCE_THRESHOLD`, `PANEL_MIN_FREQ_FOR_CLASSIFY`
- **Change models**: Set `OLLAMA_MODEL`, `OLLAMA_VALIDATION_MODEL` env vars
- **Extend taxonomy**: Run `python scripts/bootstrap_taxonomy.py --top-n 1000` to regenerate from data

## License

Internal tool for Moroccan public procurement data classification.