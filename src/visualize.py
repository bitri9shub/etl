"""
visualize.py
============
Panel-centric terminal insights from the gold parquet file.

Usage:
    python src/visualize.py
    python src/visualize.py --panel "plomberie"   # drill-down on one panel
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE    = PROJECT_ROOT / "data" / "gold" / "Article_ai_ready.parquet"

console = Console()


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def tbl(title: str, cols: list[tuple[str, str, str]]) -> Table:
    t = Table(
        title=title, box=box.ROUNDED,
        header_style="bold magenta", title_style="bold cyan",
        show_lines=True, expand=False,
    )
    for header, justify, style in cols:
        t.add_column(header, justify=justify, style=style)
    return t


# ---------------------------------------------------------------------------
# 1. Global panel overview
# ---------------------------------------------------------------------------

def table_panel_overview(df: pd.DataFrame) -> Table:
    t = tbl("Panel Coverage Overview", [
        ("Metric", "left", "white"),
        ("Value",  "right", "bold green"),
    ])
    total        = len(df)
    n_panels     = df["panel_clean"].nunique()
    top1_panel   = df["panel_clean"].value_counts().idxmax()
    top1_share   = df["panel_clean"].value_counts().max() / total * 100
    top5_share   = df["panel_clean"].value_counts().head(5).sum() / total * 100
    top10_share  = df["panel_clean"].value_counts().head(10).sum() / total * 100
    singleton_panels = (df["panel_clean"].value_counts() == 1).sum()

    rows = [
        ("Total rows",                f"{total:,}"),
        ("Unique panels",             f"{n_panels:,}"),
        ("Avg rows per panel",        f"{total/n_panels:.0f}"),
        ("Largest panel",             f"{top1_panel}  ({top1_share:.1f}%)"),
        ("Top 5 panels cover",        f"{top5_share:.1f}% of rows"),
        ("Top 10 panels cover",       f"{top10_share:.1f}% of rows"),
        ("Panels with 1 row only",    f"{singleton_panels:,}"),
        ("Panels with ≥ 1 000 rows",  f"{(df['panel_clean'].value_counts() >= 1000).sum():,}"),
        ("Panels with ≥ 10 000 rows", f"{(df['panel_clean'].value_counts() >= 10000).sum():,}"),
    ]
    for m, v in rows:
        t.add_row(m, v)
    return t


# ---------------------------------------------------------------------------
# 2. All panels ranked — full stats
# ---------------------------------------------------------------------------

def table_panels_ranked(df: pd.DataFrame, top_n: int = 30) -> Table:
    t = tbl(f"Top {top_n} Panels — Full Statistics", [
        ("#",            "right", "dim"),
        ("Panel",        "left",  "white"),
        ("Rows",         "right", "bold green"),
        ("Share %",      "right", "cyan"),
        ("Cum. %",       "right", "dim"),
        ("Avg Tokens",   "right", "yellow"),
        ("Med Tokens",   "right", "yellow"),
        ("Min Tokens",   "right", "dim"),
        ("Max Tokens",   "right", "red"),
        ("Std Tokens",   "right", "dim"),
    ])
    counts  = df["panel_clean"].value_counts().head(top_n)
    total   = len(df)
    stats   = df.groupby("panel_clean")["token_count"].agg(
        avg="mean", med="median", mn="min", mx="max", std="std"
    )
    cumulative = 0.0
    for rank, (panel, cnt) in enumerate(counts.items(), 1):
        cumulative += cnt / total * 100
        s = stats.loc[panel] if panel in stats.index else None
        t.add_row(
            str(rank),
            str(panel),
            f"{cnt:,}",
            f"{cnt/total*100:.2f}%",
            f"{cumulative:.1f}%",
            f"{s['avg']:.1f}"   if s is not None else "-",
            f"{s['med']:.0f}"   if s is not None else "-",
            f"{s['mn']:,}"      if s is not None else "-",
            f"{s['mx']:,}"      if s is not None else "-",
            f"{s['std']:.1f}"   if s is not None and not pd.isna(s['std']) else "-",
        )
    return t


# ---------------------------------------------------------------------------
# 3. Token bucket breakdown per top panel
# ---------------------------------------------------------------------------

def table_panel_token_buckets(df: pd.DataFrame, top_n: int = 15) -> Table:
    t = tbl(f"Token-Count Buckets for Top {top_n} Panels", [
        ("Panel",      "left",  "white"),
        ("Rows",       "right", "bold green"),
        ("1–10",       "right", "dim"),
        ("11–50",      "right", "cyan"),
        ("51–150",     "right", "yellow"),
        (">150",       "right", "red"),
    ])
    top_panels = df["panel_clean"].value_counts().head(top_n).index.tolist()
    sub = df[df["panel_clean"].isin(top_panels)]

    for panel in top_panels:
        p = sub[sub["panel_clean"] == panel]["token_count"]
        n = len(p)
        if n == 0:
            continue
        b1  = (p <= 10).sum()
        b2  = ((p > 10)  & (p <= 50)).sum()
        b3  = ((p > 50)  & (p <= 150)).sum()
        b4  = (p > 150).sum()
        def fmt(v): return f"{v:,} ({v/n*100:.0f}%)"
        t.add_row(str(panel), f"{n:,}", fmt(b1), fmt(b2), fmt(b3), fmt(b4))
    return t


# ---------------------------------------------------------------------------
# 4. Most verbose vs most concise panels
# ---------------------------------------------------------------------------

def table_verbosity_extremes(df: pd.DataFrame, n: int = 10) -> tuple[Table, Table]:
    stats = (
        df.groupby("panel_clean")["token_count"]
        .agg(avg="mean", count="count")
        .query("count >= 20")
        .sort_values("avg")
    )

    def build(sub, title, color):
        t = tbl(title, [
            ("#",          "right", "dim"),
            ("Panel",      "left",  "white"),
            ("Avg Tokens", "right", color),
            ("Rows",       "right", "green"),
        ])
        for rank, (panel, row) in enumerate(sub.iterrows(), 1):
            t.add_row(str(rank), str(panel), f"{row['avg']:.1f}", f"{int(row['count']):,}")
        return t

    return (
        build(stats.head(n),  f"Top {n} Most Concise Panels",  "cyan"),
        build(stats.tail(n).iloc[::-1], f"Top {n} Most Verbose Panels", "red"),
    )


# ---------------------------------------------------------------------------
# 5. Panel-level diversity (unique designation count)
# ---------------------------------------------------------------------------

def table_panel_diversity(df: pd.DataFrame, top_n: int = 20) -> Table:
    t = tbl(f"Top {top_n} Panels by Designation Diversity", [
        ("#",                  "right", "dim"),
        ("Panel",              "left",  "white"),
        ("Unique Designations","right", "bold green"),
        ("Total Rows",         "right", "cyan"),
        ("Diversity %",        "right", "yellow"),
    ])
    stats = (
        df.groupby("panel_clean")
        .agg(unique=("designation_clean", "nunique"), total=("designation_clean", "count"))
        .assign(ratio=lambda x: x["unique"] / x["total"] * 100)
        .sort_values("unique", ascending=False)
        .head(top_n)
    )
    for rank, (panel, row) in enumerate(stats.iterrows(), 1):
        t.add_row(
            str(rank), str(panel),
            f"{int(row['unique']):,}",
            f"{int(row['total']):,}",
            f"{row['ratio']:.1f}%",
        )
    return t


# ---------------------------------------------------------------------------
# 6. Drill-down: single panel
# ---------------------------------------------------------------------------

def drilldown(df: pd.DataFrame, panel_name: str) -> None:
    mask = df["panel_clean"].str.lower().str.contains(panel_name.lower(), na=False)
    sub  = df[mask]

    if sub.empty:
        console.print(f"[bold red]No rows found for panel matching:[/] '{panel_name}'")
        return

    console.print(Rule(f"[bold cyan]Drill-down: '{panel_name}'[/]"))
    console.print(f"[bold green]{len(sub):,}[/] rows across "
                  f"[bold green]{sub['panel_clean'].nunique()}[/] matching panel(s)\n")

    # Token stats
    tokens = sub["token_count"]
    t = tbl("Token Statistics", [("Metric","left","white"),("Value","right","bold green")])
    for m, v in [
        ("Rows",    f"{len(sub):,}"),
        ("Avg",     f"{tokens.mean():.1f}"),
        ("Median",  f"{tokens.median():.0f}"),
        ("Min",     f"{tokens.min():,}"),
        ("Max",     f"{tokens.max():,}"),
        ("p90",     f"{tokens.quantile(0.90):.0f}"),
        ("p95",     f"{tokens.quantile(0.95):.0f}"),
        ("Std",     f"{tokens.std():.1f}"),
    ]:
        t.add_row(m, v)
    console.print(t)
    console.print()

    # Sample designations from each quartile
    q25, q75 = tokens.quantile(0.25), tokens.quantile(0.75)
    s = tbl("Sample Designations (short / medium / long)", [
        ("Bucket", "left", "cyan"),
        ("Tokens", "right", "yellow"),
        ("Designation", "left", "white"),
    ])
    for label, cond in [
        ("Short  (≤ p25)", sub[tokens <= q25]),
        ("Medium (p25–p75)", sub[(tokens > q25) & (tokens <= q75)]),
        ("Long   (> p75)", sub[tokens > q75]),
    ]:
        sample = cond.head(3)
        for _, row in sample.iterrows():
            text = str(row["designation_clean"])
            text = text[:120] + "…" if len(text) > 120 else text
            s.add_row(label, str(row["token_count"]), text)
            label = ""  # only print label once per bucket
    console.print(s)


# ---------------------------------------------------------------------------
# 7. Canonical consolidation stats
# ---------------------------------------------------------------------------

def table_canonical_overview(df: pd.DataFrame) -> Table:
    t = tbl("Canonical Consolidation", [
        ("Metric", "left", "white"),
        ("Value",  "right", "bold green"),
    ])
    total        = len(df)
    n_raw        = df["panel_clean"].nunique()
    n_canonical  = df["panel_canonical"].nunique()
    ratio        = n_canonical / n_raw * 100 if n_raw else 0
    top1         = df["panel_canonical"].value_counts().idxmax()
    top1_share   = df["panel_canonical"].value_counts().max() / total * 100
    top5_share   = df["panel_canonical"].value_counts().head(5).sum() / total * 100
    singleton    = (df["panel_canonical"].value_counts() == 1).sum()
    n_unmatched  = (df["panel_canonical"] == df["panel_clean"]).sum()

    rows = [
        ("Total rows",                       f"{total:,}"),
        ("Unique panel_clean",               f"{n_raw:,}"),
        ("Unique panel_canonical",           f"{n_canonical:,}"),
        ("Ratio canon/raw",                  f"{ratio:.1f}%"),
        ("Largest canonical",                f"{top1}  ({top1_share:.1f}%)"),
        ("Top 5 canonicals cover",           f"{top5_share:.1f}% of rows"),
        ("Canonicals with 1 row",            f"{singleton:,}"),
        ("Unmatched (identity fallback)",    f"{n_unmatched:,}"),
    ]
    for m, v in rows:
        t.add_row(m, v)
    return t


def table_canonical_ranked(df: pd.DataFrame, top_n: int = 20) -> Table:
    t = tbl(f"Top {top_n} Canonical Groups", [
        ("#",              "right", "dim"),
        ("Canonical Name", "left",  "white"),
        ("Rows",           "right", "bold green"),
        ("Share %",        "right", "cyan"),
        ("Cum. %",         "right", "dim"),
        ("Raw Panels Merged","right","yellow"),
        ("Avg Tokens",     "right", "yellow"),
        ("L1",             "left",  "dim"),
        ("L2",             "left",  "dim"),
        ("L3",             "left",  "dim"),
        ("L4",             "left",  "dim"),
        ("L5",             "left",  "dim"),
    ])
    counts = df["panel_canonical"].value_counts().head(top_n)
    total  = len(df)
    merged = df.groupby("panel_canonical")["panel_clean"].nunique()
    tokens = df.groupby("panel_canonical")["token_count"].mean()
    l1_map = df["panel_canonical"].map(df["category_l1"])
    l2_map = df["panel_canonical"].map(df["category_l2"])
    l3_map = df["panel_canonical"].map(df["category_l3"])
    l4_map = df["panel_canonical"].map(df["category_l4"])
    l5_map = df["panel_canonical"].map(df["category_l5"])

    cumulative = 0.0
    for rank, (canon, cnt) in enumerate(counts.items(), 1):
        cumulative += cnt / total * 100
        t.add_row(
            str(rank), str(canon), f"{cnt:,}",
            f"{cnt/total*100:.2f}%", f"{cumulative:.1f}%",
            f"{int(merged.get(canon, 1)):,}",
            f"{tokens.get(canon, 0):.1f}",
            str(l1_map.get(canon, "")),
            str(l2_map.get(canon, "")),
            str(l3_map.get(canon, "")),
            str(l4_map.get(canon, "")),
            str(l5_map.get(canon, "")),
        )
    return t


# ---------------------------------------------------------------------------
# 8. Taxonomy coverage & tree
# ---------------------------------------------------------------------------

def table_taxonomy_coverage(df: pd.DataFrame) -> Table:
    t = tbl("Taxonomy Coverage", [
        ("Level", "left", "white"),
        ("Rows Classified", "right", "bold green"),
        ("% of Total", "right", "cyan"),
    ])
    total = len(df)

    # Classification status: has a non-identity canonical
    has_canon = df["panel_canonical"].notna()
    n_classified = (has_canon & (df["panel_canonical"] != df["panel_clean"])).sum()
    n_identity = total - n_classified

    t.add_row("Canonical (non-identity)", f"{n_classified:,}", f"{n_classified/total*100:.1f}%")
    t.add_row("Identity fallback", f"{n_identity:,}", f"{n_identity/total*100:.1f}%")

    if "category_l1" in df.columns:
        has_l1 = df["category_l1"].notna() & (df["category_l1"] != "Non classé")
        n_l1 = has_l1.sum()
        t.add_row("Mapped to L1 category", f"{n_l1:,}", f"{n_l1/total*100:.1f}%")
        if "category_l2" in df.columns:
            has_l2 = df["category_l2"].notna() & (df["category_l2"] != "Non classé")
            n_l2 = has_l2.sum()
            t.add_row("Mapped to L2 category", f"{n_l2:,}", f"{n_l2/total*100:.1f}%")
            if "category_l3" in df.columns:
                has_l3 = df["category_l3"].notna() & (df["category_l3"] != "Non classé")
                n_l3 = has_l3.sum()
                t.add_row("Mapped to L3 category", f"{n_l3:,}", f"{n_l3/total*100:.1f}%")
                if "category_l4" in df.columns:
                    has_l4 = df["category_l4"].notna() & (df["category_l4"] != "Non classé")
                    n_l4 = has_l4.sum()
                    t.add_row("Mapped to L4 category", f"{n_l4:,}", f"{n_l4/total*100:.1f}%")
                    if "category_l5" in df.columns:
                        has_l5 = df["category_l5"].notna() & (df["category_l5"] != "Non classé")
                        n_l5 = has_l5.sum()
                        t.add_row("Mapped to L5 category", f"{n_l5:,}", f"{n_l5/total*100:.1f}%")

    return t


def table_taxonomy_tree(df: pd.DataFrame) -> Table:
    if "category_l1" not in df.columns or "category_l2" not in df.columns:
        t = Table(title="Taxonomy Tree (not available — run taxonomy.py --enrich)",
                  box=box.ROUNDED)
        t.add_column("Info", style="yellow")
        t.add_row("category_l1 / category_l2 columns not found in gold")
        return t

    t = tbl("Taxonomy Tree — Rows per L1 > L2 > L3 > L4 > L5", [
        ("L1", "left", "bold yellow"),
        ("L2", "left", "cyan"),
        ("L3", "left", "green"),
        ("L4", "left", "yellow"),
        ("L5", "left", "dim"),
        ("Rows", "right", "bold green"),
        ("%", "right", "dim"),
    ])
    total = len(df)
    stats = df.groupby(["category_l1", "category_l2", "category_l3", "category_l4", "category_l5"]).size().sort_values(ascending=False)

    shown_l1 = set()
    shown_l2 = set()
    shown_l3 = set()
    shown_l4 = set()

    for (l1, l2, l3, l4, l5), cnt in stats.items():
        label_l1 = l1 if l1 not in shown_l1 else ""
        shown_l1.add(l1)
        label_l2 = l2 if l2 not in shown_l2 else ""
        shown_l2.add(l2)
        label_l3 = l3 if l3 not in shown_l3 else ""
        shown_l3.add(l3)
        label_l4 = l4 if l4 not in shown_l4 else ""
        shown_l4.add(l4)
        t.add_row(label_l1, label_l2, label_l3, label_l4, str(l5) if l5 else "", f"{cnt:,}", f"{cnt/total*100:.1f}%")

    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(panel_filter: str | None = None) -> None:
    if not GOLD_FILE.exists():
        console.print(f"[bold red]ERROR:[/] Gold file not found:\n  {GOLD_FILE}")
        sys.exit(1)

    console.print(Rule("[bold cyan]Gold Dataset — Panel-Centric Insights[/]"))
    console.print(f"[dim]Source:[/] {GOLD_FILE}\n")

    df = pd.read_parquet(GOLD_FILE)
    console.print(f"[bold green]Loaded[/] {len(df):,} rows | "
                  f"{df['panel_clean'].nunique():,} unique panels\n")

    # ---- Drill-down mode ----
    if panel_filter:
        drilldown(df, panel_filter)
        return

    # ---- Full report ----
    console.print(table_panel_overview(df));          console.print()
    console.print(table_panels_ranked(df, top_n=30)); console.print()
    console.print(table_panel_token_buckets(df, top_n=15)); console.print()

    concise, verbose = table_verbosity_extremes(df, n=10)
    console.print(Columns([concise, verbose], equal=False, expand=False))
    console.print()

    console.print(table_panel_diversity(df, top_n=20)); console.print()

    # ---- Canonical / Taxonomy section ----
    if "panel_canonical" in df.columns:
        console.print(Rule("[bold cyan]Canonical & Taxonomy[/]"))
        console.print(table_canonical_overview(df)); console.print()
        console.print(table_canonical_ranked(df, top_n=20)); console.print()
        console.print(table_taxonomy_coverage(df)); console.print()
        console.print(table_taxonomy_tree(df)); console.print()

    console.print(Rule("[dim]Done — use --panel <name> for a drill-down[/]"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panel-centric gold insights.")
    parser.add_argument("--panel", default=None,
                        help="Partial panel name to drill down on (case-insensitive).")
    args = parser.parse_args()
    main(panel_filter=args.panel)
