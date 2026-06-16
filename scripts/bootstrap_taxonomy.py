"""
bootstrap_taxonomy.py
======================
One-time bootstrap: reads gold, uses Ollama to group top panels
into a taxonomy, writes taxonomy.json and classification cache.

Usage:
    python scripts/bootstrap_taxonomy.py --top-n 500
    python scripts/bootstrap_taxonomy.py --top-n 1000 --no-cache
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
from rich.console import Console
from rich.rule import Rule

from config.settings import GOLD_FILE, TAXONOMY_FILE, CLASSIFIER_CACHE
from taxonomy import bootstrap_taxonomy, report_taxonomy

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Bootstrap taxonomy from gold data")
    parser.add_argument("--top-n", type=int, default=500,
                        help="Number of top panels to use (default: 500)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Clear existing classification cache before starting")
    args = parser.parse_args()

    if not GOLD_FILE.exists():
        console.print(f"[red]Gold file not found: {GOLD_FILE}[/]")
        sys.exit(1)

    console.print(Rule("[bold cyan]Taxonomy Bootstrap[/]"))
    console.print(f"Model     : qwen2.5:3b-instruct (configurable via OLLAMA_MODEL env)")
    console.print(f"Top panels: {args.top_n}")
    console.print(f"Gold file : {GOLD_FILE}")
    console.print()

    if args.no_cache:
        if CLASSIFIER_CACHE.exists():
            CLASSIFIER_CACHE.unlink()
            console.print("[yellow]Classification cache cleared.[/]")

    df = pd.read_parquet(GOLD_FILE)
    console.print(f"Loaded {len(df):,} gold rows with "
                  f"{df['panel_clean'].nunique():,} unique panels\n")

    tax = bootstrap_taxonomy(df, top_n=args.top_n)

    console.print()
    console.print(Rule("[bold green]Done![/]"))
    console.print(f"Taxonomy written to: {TAXONOMY_FILE}")
    console.print(f"{len(tax['patterns'])} patterns across {len(tax['tree'])} L1 categories")

    console.print()
    report_taxonomy(df)


if __name__ == "__main__":
    main()
