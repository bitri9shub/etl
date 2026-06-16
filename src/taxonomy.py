"""
taxonomy.py
============
Standalone taxonomy enrichment and bootstrap for gold data.

Usage:
    python src/taxonomy.py --enrich       # classify unmatched panels in gold
    python src/taxonomy.py --bootstrap    # bootstrap taxonomy.json from top panels
    python src/taxonomy.py --report       # print taxonomy tree summary
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich import box

from config.settings import (
    GOLD_FILE, TAXONOMY_FILE, TAXONOMY_MAP_FILE, TAXONOMY_DIR,
    CLASSIFIER_CACHE, PANEL_MIN_FREQ_FOR_CLASSIFY
)
from main import load_taxonomy_patterns, canonicalize_panel, normalize_panel

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_unmatched_panels(gold_df: pd.DataFrame) -> pd.Series:
    """Return panel_clean values where canonicalization fell back to identity."""
    patterns = load_taxonomy_patterns()
    if not patterns:
        return gold_df["panel_clean"].dropna()

    def is_unmatched(p):
        if not p:
            return True
        for entry in patterns:
            if __import__("re").search(entry["regex"], p):
                return False
        return True

    return gold_df["panel_clean"].dropna().loc[
        gold_df["panel_clean"].dropna().apply(is_unmatched)
    ]


# ---------------------------------------------------------------------------
# Enrich: classify unmatched panels via Ollama
# ---------------------------------------------------------------------------

def enrich_gold(gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds panels in gold that weren't matched by rules,
    classifies them via Ollama, and adds category_l1 / category_l2 columns.
    Returns a copy of gold with added taxonomy columns.
    """
    from classifier import TaxonomyClassifier

    df = gold_df.copy()

    # Find unmatched unique panels with frequency >= threshold
    unmatched = find_unmatched_panels(df)
    freq = unmatched.value_counts()
    to_classify = freq[freq >= PANEL_MIN_FREQ_FOR_CLASSIFY].index.tolist()

    if not to_classify:
        console.print("[yellow]No unmatched panels to classify.[/]")
        df["category_l1"] = df["panel_canonical"].apply(
            lambda p: _canonical_to_l1(p) if p else "Non classé"
        )
        df["category_l2"] = df["panel_canonical"].apply(
            lambda p: _canonical_to_l2(p) if p else "Non classé"
        )
        df["category_l3"] = df["panel_canonical"].apply(
            lambda p: _canonical_to_l3(p) if p else "Non classé"
        )
        df["category_l4"] = df["panel_canonical"].apply(
            lambda p: _canonical_to_l4(p) if p else "Non classé"
        )
        df["category_l5"] = df["panel_canonical"].apply(
            lambda p: _canonical_to_l5(p) if p else "Non classé"
        )
        return df

    console.print(f"[cyan]Classifying {len(to_classify):,} unmatched panel values...[/]")
    classifier = TaxonomyClassifier()

    classifications = classifier.classify_batch(to_classify)

    # Build a lookup: panel_clean → (canonical, l1, l2, l3, l4, l5)
    lookup = {}
    for panel, result in classifications.items():
        canonical = result.get("canonical") or panel
        l1 = result.get("l1", "Non classé")
        l2 = result.get("l2", "Non classé")
        l3 = result.get("l3", "Non classé")
        l4 = result.get("l4", "Non classé")
        l5 = result.get("l5", "Non classé")
        lookup[panel] = (canonical, l1, l2, l3, l4, l5)

    # Update panel_canonical where it was identity fallback
    mask = df["panel_clean"].isin(lookup)
    df.loc[mask, "panel_canonical"] = df.loc[mask, "panel_clean"].map(
        lambda p: lookup[p][0]
    )

    # Add taxonomy columns
    def get_l1(p):
        if not p:
            return "Non classé"
        if p in lookup:
            return lookup[p][1]
        return _canonical_to_l1(p)

    def get_l2(p):
        if not p:
            return "Non classé"
        if p in lookup:
            return lookup[p][2]
        return _canonical_to_l2(p)

    def get_l3(p):
        if not p:
            return "Non classé"
        if p in lookup:
            return lookup[p][3]
        return _canonical_to_l3(p)

    def get_l4(p):
        if not p:
            return "Non classé"
        if p in lookup:
            return lookup[p][4]
        return _canonical_to_l4(p)

    def get_l5(p):
        if not p:
            return "Non classé"
        if p in lookup:
            return lookup[p][5]
        return _canonical_to_l5(p)

    df["category_l1"] = df["panel_canonical"].apply(get_l1)
    df["category_l2"] = df["panel_canonical"].apply(get_l2)
    df["category_l3"] = df["panel_canonical"].apply(get_l3)
    df["category_l4"] = df["panel_canonical"].apply(get_l4)
    df["category_l5"] = df["panel_canonical"].apply(get_l5)

    console.print(f"[green]Classified {len(lookup):,} panels via Ollama.[/]")
    return df


def _canonical_to_l1(canonical: str) -> str:
    """Look up L1 from taxonomy patterns."""
    patterns = load_taxonomy_patterns()
    for entry in patterns:
        if entry["canonical"] == canonical:
            return entry.get("l1", "Non classé")
    return "Non classé"


def _canonical_to_l2(canonical: str) -> str:
    patterns = load_taxonomy_patterns()
    for entry in patterns:
        if entry["canonical"] == canonical:
            return entry.get("l2", "Non classé")
    return "Non classé"


def _canonical_to_l3(canonical: str) -> str:
    patterns = load_taxonomy_patterns()
    for entry in patterns:
        if entry["canonical"] == canonical:
            return entry.get("l3", "Non classé")
    return "Non classé"


def _canonical_to_l4(canonical: str) -> str:
    patterns = load_taxonomy_patterns()
    for entry in patterns:
        if entry["canonical"] == canonical:
            return entry.get("l4", "Non classé")
    return "Non classé"


def _canonical_to_l5(canonical: str) -> str:
    patterns = load_taxonomy_patterns()
    for entry in patterns:
        if entry["canonical"] == canonical:
            return entry.get("l5", "Non classé")
    return "Non classé"


# ---------------------------------------------------------------------------
# Bootstrap: generate taxonomy.json from top-N panels using Ollama
# ---------------------------------------------------------------------------

def bootstrap_taxonomy(gold_df: pd.DataFrame, top_n: int = 500) -> dict:
    """
    Takes the top-N most frequent panel_clean values from gold,
    uses Ollama to group them into a taxonomy tree,
    and writes the result to TAXONOMY_FILE.
    """
    from classifier import TaxonomyClassifier

    freq = gold_df["panel_clean"].value_counts()
    top_panels = freq.head(top_n).index.tolist()

    console.print(f"[cyan]Bootstrapping taxonomy from top {len(top_panels)} panels...[/]")

    classifier = TaxonomyClassifier()
    groups = classifier.bootstrap_from_panels(top_panels, batch_size=100)

    patterns = []
    all_l1 = set()
    tree = {}

    for group in groups:
        canon = group.get("canonical", "")
        l1 = group.get("l1", "Non classé")
        l2 = group.get("l2", "Non classé")
        l3 = group.get("l3", "Non classé")
        l4 = group.get("l4", "Non classé")
        l5 = group.get("l5", "Non classé")
        panels = group.get("panels", [])

        if not canon or not panels:
            continue

        all_l1.add(l1)

        # Build regex from panel names
        escaped = [re.escape(p) for p in panels]
        regex = "|".join(escaped)

        patterns.append({
            "regex": regex,
            "canonical": canon,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "l4": l4,
            "l5": l5,
        })

        if l1 not in tree:
            tree[l1] = {"description": "", "children": []}
        if l2 not in tree[l1]["children"]:
            tree[l1]["children"].append(l2)

    # Add "Non classé" if not present
    if "Non classé" not in tree:
        tree["Non classé"] = {"description": "Panels non classifiés", "children": ["Non classé"]}

    taxonomy = {
        "version": "1.0",
        "description": f"Auto-bootstrapped from top {top_n} panels",
        "default_canonical": None,
        "patterns": patterns,
        "tree": tree,
    }

    # Write
    TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TAXONOMY_FILE, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Taxonomy written to {TAXONOMY_FILE}[/]")
    console.print(f"  {len(patterns)} patterns")
    console.print(f"  {len(tree)} L1 categories")
    return taxonomy


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report_taxonomy(gold_df: pd.DataFrame | None = None):
    """Print taxonomy tree summary."""

    if not TAXONOMY_FILE.exists():
        console.print("[red]No taxonomy file found.[/]")
        return

    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        tax = json.load(f)

    patterns = tax.get("patterns", [])
    tree = tax.get("tree", {})

    # Overview
    t = Table(title="Taxonomy Overview", box=box.ROUNDED,
              header_style="bold magenta", title_style="bold cyan")
    t.add_column("Metric", justify="left", style="white")
    t.add_column("Value", justify="right", style="bold green")
    t.add_row("L1 Categories", str(len(tree)))
    t.add_row("L2 Categories", str(sum(len(v.get("children", [])) for v in tree.values())))
    t.add_row("L3 Categories", str(sum(len(v.get("children", [])) for v in tree.values() if isinstance(v.get("children"), list) for c in v.get("children", []) if isinstance(c, dict))))
    t.add_row("Regex Patterns", str(len(patterns)))
    t.add_row("Version", tax.get("version", "?"))
    console.print(t)

    # Tree view
    t2 = Table(title="Taxonomy Tree", box=box.ROUNDED,
               header_style="bold cyan", title_style="bold cyan")
    t2.add_column("L1", justify="left", style="bold yellow")
    t2.add_column("L2", justify="left", style="cyan")
    t2.add_column("L3", justify="left", style="green")
    t2.add_column("L4", justify="left", style="yellow")
    t2.add_column("L5", justify="left", style="dim")
    t2.add_column("Rows", justify="right", style="bold green")
    t2.add_column("%", justify="right", style="dim")

    for l1, info in sorted(tree.items()):
        children = info.get("children", [])
        for i, l2 in enumerate(children):
            label_l1 = l1 if i == 0 else ""
            t2.add_row(label_l1, str(l2) if l2 else "", "", "", "", "-", "-")  # Placeholder for L3-L5

    # Coverage (if gold provided)
    if gold_df is not None and "panel_canonical" in gold_df.columns:
        console.print()
        console.print(Rule("[bold cyan]Coverage[/]"))
        total = len(gold_df)
        classified = gold_df["panel_canonical"].notna() & (gold_df["panel_canonical"] != "")
        n_classified = classified.sum()
        n_unclassified = total - n_classified
        pct = n_classified / total * 100

        t3 = Table(box=box.ROUNDED, header_style="bold magenta")
        t3.add_column("Category", justify="left", style="white")
        t3.add_column("Rows", justify="right", style="bold green")
        t3.add_column("%", justify="right", style="cyan")
        t3.add_row("Classified (canonical ≠ identity)", f"{n_classified:,}", f"{pct:.1f}%")
        t3.add_row("Unclassified (identity fallback)", f"{n_unclassified:,}", f"{100 - pct:.1f}%")
        console.print(t3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Taxonomy enrichment and bootstrap")
    parser.add_argument("--enrich", action="store_true", help="Classify unmatched panels via Ollama")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap taxonomy.json from top panels")
    parser.add_argument("--report", action="store_true", help="Print taxonomy tree summary")
    parser.add_argument("--top-n", type=int, default=500, help="Top N panels for bootstrap")
    parser.add_argument("--output", type=str, default=str(TAXONOMY_MAP_FILE),
                        help="Output path for taxonomy map")
    args = parser.parse_args()

    if not args.enrich and not args.bootstrap and not args.report:
        parser.print_help()
        return

    # Load gold if needed
    gold_df = None
    if args.enrich or args.report:
        if not GOLD_FILE.exists():
            console.print(f"[red]Gold file not found: {GOLD_FILE}[/]")
            sys.exit(1)
        gold_df = pd.read_parquet(GOLD_FILE)
        console.print(f"[green]Loaded {len(gold_df):,} gold rows[/]")

    if args.report:
        report_taxonomy(gold_df)

    if args.bootstrap:
        if gold_df is None:
            gold_df = pd.read_parquet(GOLD_FILE)
        bootstrap_taxonomy(gold_df, top_n=args.top_n)

    if args.enrich:
        enriched = enrich_gold(gold_df)
        TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(args.output, index=False, compression="zstd")
        console.print(f"[green]Taxonomy-enriched gold saved: {args.output}[/]")

        # Also save the lookup map
        map_df = enriched[["panel_clean", "panel_canonical", "category_l1", "category_l2"]].drop_duplicates()
        map_path = TAXONOMY_DIR / "taxonomy_map.parquet"
        map_df.to_parquet(map_path, index=False, compression="zstd")
        console.print(f"[green]Taxonomy map saved: {map_path} ({len(map_df):,} entries)[/]")


if __name__ == "__main__":
    main()
