import json
import uuid
import time
import re
import unicodedata
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from config.settings import (
    RAW_DIR,  RAW_FILE,  BRONZE_DIR,  CHUNK_SIZE,
    CSV_ENCODING, CSV_SEPARATOR, REQUIRED_COLS, STATE_FILE,
    AUDIT_DIR, MAX_TOKENS, SILVER_FILE, GOLD_DIR, GOLD_FILE,
    SILVER_DIR, BOILERPLATE_PATTERNS, NULL_MARKERS, REJECTED_FILE,
    TAXONOMY_FILE
)

# ============================================================
# OPERATIONS
# ============================================================

# TEXT TRANSFORMATION
def to_lowercase(text: str) -> str:
    return text.lower()

def remove_accents(text: str) -> str: 
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    
def remove_punctuation(text: str) -> str:
    # Convertit tous les exposants/indices Unicode en leur équivalent ASCII
    result = []
    for char in text:
        name = unicodedata.name(char, "")
        if "SUPERSCRIPT" in name:
            # Ex: "SUPERSCRIPT TWO" → "2"
            word = name.split()[-1]  # prend le dernier mot "TWO", "THREE"...
            digit = str(unicodedata.digit(char, None))
            result.append(digit if digit != "None" else char)
        else:
            result.append(char)
    text = "".join(result)
    return re.sub(r"[^\w\s°/\.\-]", " ", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def standardize_units(text: str) -> str:
    # Unités standard : chiffre + espace optionnel + unité
    UNITS = ["gb", "tb", "mb", "mm", "cm", "kg", "kw", "mpa", "bar", "m", "g", "l", "w"]
    
    for unit in UNITS:
        text = re.sub(rf"\b(\d+)\s*{unit}\b", rf"\g<1>{unit}", text, flags=re.IGNORECASE)
    
    # Cas spéciaux : préfixe + espace optionnel + chiffre
    PREFIXES = ["dn", "pn"]
    for prefix in PREFIXES:
        text = re.sub(rf"\b{prefix}\s*(\d+)\b", rf"{prefix}\g<1>", text, flags=re.IGNORECASE)
    
    # Cas particulier : "TO" → "tb"
    text = re.sub(r"\b(\d+)\s*to\b", r"\g<1>tb", text, flags=re.IGNORECASE)
    
    return text   

def remove_boilerplate(text: str, patterns: dict[str, str] = BOILERPLATE_PATTERNS) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    # Étape 1 : Application séquentielle des expressions régulières
    for name, pattern in patterns.items():
        # Remplacement par un espace pour éviter de coller deux mots accidentellement
        text = re.sub(pattern, " ", text)
    
    # Étape 2 : Nettoyage post-regex des résidus textuels (Espaces et ponctuations orphelines)
    # 1. Supprime les puces d'énumération isolées (*)
    text = re.sub(r'\*\s*', ' ', text)
    # 2. Nettoie les espaces multiples provoqués par les suppressions répétées
    text = re.sub(r'\s+', ' ', text)
    # 3. Corrige la ponctuation orpheline (ex: "PN16 ; , " -> "PN16;")
    text = re.sub(r'\s+([,;.:\-])\s*', r'\1 ', text)
    # 4. Supprime la ponctuation qui se retrouverait coincée tout à la fin du texte
    text = re.sub(r'^[,\s;.:\-]+|[,\s;.:\-]+$', '', text)
    
    return text.strip()

def is_null_marker(text: str, null_markers: list[str] = NULL_MARKERS) -> bool:
    """Returns True if the text is effectively empty/null."""
    return text.strip().lower() in null_markers

def normalize(text: str) -> str | None:
    """
    For normalization pipeline for a single text value.
    Returns None if the text is null marker after cleaning
    """
    if not isinstance(text, str):
        return None
    
    text = to_lowercase(text)
    text = remove_accents(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    text = remove_boilerplate(text)
    text = standardize_units(text)
    text = normalize_whitespace(text)

    if is_null_marker(text):
        return None
    return text

# PANEL-SPECIFIC NORMALIZATION

LEADING_VERBS = [
    r"^achats?\s+d(?:\s+|e\s+)",
    r"^acquisitions?\s+d(?:\s+|e\s+)",
    r"^locations?\s+d(?:\s+|e\s+)",
    r"^remplacements?\s+d(?:\s+|e\s+)",
    r"^installations?\s+d(?:\s+|e\s+)",
    r"^maintenances?\s+d(?:\s+|e\s+)",
    r"^entretiens?\s+d(?:\s+|e\s+)",
    r"^aménagements?\s+d(?:\s+|e\s+)",
    r"^amenagements?\s+d(?:\s+|e\s+)",
    r"^mise\s+en\s+(?:place|œuvre|oeuvre|service)\s+d(?:\s+|e\s+)",
    r"^fourniture\s+et\s+pose\s+d(?:\s+|e\s+)",
    r"^prestations?\s+de\s+",
    r"^réparations?\s+de\s+",
    r"^reparations?\s+de\s+",
    r"^rénovations?\s+de\s+",
    r"^renovations?\s+de\s+",
]

LEADING_DETERMINERS = [
    r"^(?:un|une|des?|du|de\s+la)\s+",
]

TRAILING_NOISE = [
    r"\s+\d+[\d\s,]*\s*(kg|g|l|m|cm|mm|kw|w|€|euros?|unité|unités|piece|pieces|pièce|pièces|lot|lots|ml|cl|dl)\.?\s*$",
    r"\s+(?:etc|etc\.|…|\.\.\.)\s*$",
    r"\s+(environ|approx|environnement)\s*[\d\s]*$",
    r"\s+(?:y\s+compris|y/c|compris)\s*$",
    r"\s+(?:hors\s+taxe|ht|ttc)\s*$",
    r"\s+\(.*\)\s*$",
    r"\s+\[.*\]\s*$",
]


def strip_leading_verbs(text: str) -> str:
    """Strip leading procurement verb phrases from panel text."""
    for pattern in LEADING_VERBS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    for pattern in LEADING_DETERMINERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_trailing_noise(text: str) -> str:
    """Strip trailing noise words, units, and quantities from panel text."""
    for pattern in TRAILING_NOISE:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_panel(text: str) -> str | None:
    """
    Panel-specific normalization pipeline.
    Same base as normalize() + strips leading verbs and trailing noise.
    """
    if not isinstance(text, str):
        return None

    text = to_lowercase(text)
    text = remove_accents(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    text = remove_boilerplate(text)
    text = strip_leading_verbs(text)
    text = strip_trailing_noise(text)
    text = standardize_units(text)
    text = normalize_whitespace(text)

    if is_null_marker(text):
        return None
    return text


_TAXONOMY_PATTERNS = None


def load_taxonomy_patterns() -> list[dict]:
    global _TAXONOMY_PATTERNS
    if _TAXONOMY_PATTERNS is not None:
        return _TAXONOMY_PATTERNS
    if not TAXONOMY_FILE.exists():
        _TAXONOMY_PATTERNS = []
        return _TAXONOMY_PATTERNS
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        tax = json.load(f)
    _TAXONOMY_PATTERNS = tax.get("patterns", [])
    return _TAXONOMY_PATTERNS


def canonicalize_panel(panel_clean: str) -> tuple[str, str, str, str, str]:
    """
    Apply regex rules from taxonomy.json to map panel_clean → canonical group.
    Returns (panel_canonical, category_l1, category_l2, category_l3, category_l4, category_l5).
    Falls back to (panel_clean, "Non classé", "Non classé", "Non classé", "Non classé", "Non classé") if no rule matches.
    """
    if not isinstance(panel_clean, str) or not panel_clean:
        return (panel_clean if isinstance(panel_clean, str) else "", "Non classé", "Non classé", "Non classé", "Non classé", "Non classé")

    patterns = load_taxonomy_patterns()
    for entry in patterns:
        if re.search(entry["regex"], panel_clean):
            return (
                entry["canonical"],
                entry.get("l1", "Non classé"),
                entry.get("l2", "Non classé"),
                entry.get("l3", "Non classé"),
                entry.get("l4", "Non classé"),
                entry.get("l5", "Non classé"),
            )

    return (panel_clean, "Non classé", "Non classé", "Non classé", "Non classé", "Non classé")


# REPORT
def generate_report() -> dict:
    """
    Builds a quality report dict from the run log and gold/silver stats.
    """
    import csv
    from config.settings import RUN_LOG_FILE

    report = {
        "silver": silver_stats() if silver_exists() else {"exists": False},
        "gold":   gold_stats()   if gold_exists()   else {"exists": False},
        "chunks": [],
    }

    if RUN_LOG_FILE.exists():
        with open(RUN_LOG_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if rows:
            total_rows    = sum(int(r["total_rows"])        for r in rows)
            valid_rows    = sum(int(r["valid_rows"])        for r in rows)
            rejected_rows = sum(int(r["rejected_rows"])     for r in rows)
            written_rows  = sum(int(r["written_to_silver"]) for r in rows)
            total_dur     = sum(float(r["duration_s"])      for r in rows)

            report["run_summary"] = {
                "chunks_processed":   len(rows),
                "total_rows":         total_rows,
                "valid_rows":         valid_rows,
                "rejected_rows":      rejected_rows,
                "rejection_rate_pct": round(rejected_rows / total_rows * 100, 2) if total_rows else 0,
                "written_to_silver":  written_rows,
                "total_duration_s":   round(total_dur, 2),
            }
            report["chunks"] = rows

    return report

def print_report(report: dict) -> None:
    """Prints a human-readable quality report to the console."""
    print(f"\n{'='*60}")
    print("  QUALITY REPORT")
    print(f"{'='*60}")
    summary = report.get("run_summary", {})
    if summary:
        print(f"  Chunks processed : {summary.get('chunks_processed', 'N/A')}")
        print(f"  Total rows       : {summary.get('total_rows', 0):,}")
        print(f"  Valid rows       : {summary.get('valid_rows', 0):,}")
        print(f"  Rejected rows    : {summary.get('rejected_rows', 0):,}  "
              f"({summary.get('rejection_rate_pct', 0)}%)")
        print(f"  Written to silver: {summary.get('written_to_silver', 0):,}")
        print(f"  Total duration   : {summary.get('total_duration_s', 0)}s")
    else:
        print("  (no run log data available)")
    gold = report.get("gold", {})
    if gold.get("exists"):
        print(f"\n  Gold rows        : {gold.get('total_rows', 0):,}")
        print(f"  Avg tokens       : {gold.get('avg_tokens', 'N/A')}")
        print(f"  Token range      : {gold.get('min_tokens', 'N/A')} - {gold.get('max_tokens', 'N/A')}")
    print(f"{'='*60}\n")

def save_report(report: dict) -> None:
    """Saves the quality report as a timestamped JSON file in AUDIT_DIR."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = AUDIT_DIR / f"report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    -> report saved: {report_path.name}")

# RAW / CHUNKS
def get_raw_files() -> list[Path]:
    """
    Returns all csv.gz files found in a raw directory.
    """
    return sorted(RAW_DIR.glob('*.csv.gz'))

def get_existing_chunks(bronze_dir: Path) -> set[str]:
    """
    Return filenames of already processed chunks.
    """
    return {
        f.name for f in bronze_dir.glob("chunk_*.parquet")
    }

def build_chunk_filename(start: int, end: int) -> str:
    """
    e.g. chunk_0000001_0100000.parquet
    """
    return f"chunk_{start:07d}_{end:07d}.parquet"

def chunk_raw_file(
        raw_file: Path = RAW_FILE,
        bronze_dir: Path = BRONZE_DIR,
) -> list[Path]:
    """
    Reads raw csv.gz in chunks of CHUNK_SIZE rows.
    Writes each chunk as a Parquet file in BRONZE_DIR.
    Skips chunk that already exist (resumable).
    Returns list of written chunk paths.
    """
    
    bronze_dir.mkdir(parents=True, exist_ok=True)
    existing = get_existing_chunks(bronze_dir)
    written = []
    row_start = 1

    reader = pd.read_csv(
        raw_file,
        sep = CSV_SEPARATOR,
        encoding = CSV_ENCODING,
        chunksize = CHUNK_SIZE,
        dtype = str,
        keep_default_na = False, # don't auto-convert "-" to NaN
        quotechar = '"', # handles multiline designation fields
        engine = 'python' # more robust than C parser for messy CSVs
    )

    for chunk_df in tqdm(reader, desc='Chunking raw file'):
        chunk_df: pd.DataFrame
        row_end = row_start + len(chunk_df) - 1
        chunk_name = build_chunk_filename(row_start, row_end)
        chunk_path = bronze_dir / chunk_name

        if chunk_name in existing:
            print(f'    -> skip (exists): {chunk_name}')
            row_start = row_end + 1
            continue

        missing = [c for c in REQUIRED_COLS if c not in chunk_df.columns]
        if missing:
            raise ValueError(f'Missing columns in chunk: {missing}')
        
        chunk_df.to_parquet(chunk_path, index=False, compression="zstd")
        written.append(chunk_path)
        print(f"  ↳ wrote: {chunk_name}  ({len(chunk_df):,} rows)")

        row_start = row_end + 1

    return written

def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies normalization to a bronze chunk DataFrame.
    Adds designation_clean and panel_clean columns
    """
    df = df.copy()
    df["designation_clean"] = df["designation"].apply(normalize)
    df["panel_clean"] = df["panel"].apply(normalize_panel)

    return df

def validate_chunk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates a normalized chunk.
    Returns (valid_df, rejected_df).
    rejected_df has an extra 'rejected_reasons' column.
    """
    if df.empty:
        df_copy = df.copy()
        df_copy["rejected_reasons"] = pd.Series(dtype=str)
        return df.reset_index(drop=True), df_copy.reset_index(drop=True)
    # checks id
    id_str = df["id"].fillna("").astype(str)
    id_missing = df["id"].isna() | (id_str.str.strip() == "")

    # checks designation
    des_str = df["designation_clean"].fillna("None").astype(str)
    des_missing = df["designation_clean"].isna() | (des_str.str.strip() == "")

    # Token count check
    token_counts = des_str.str.split().str.len()
    too_short = token_counts < 1

    # Panel check (non-blocking)
    panel_str = df["panel_clean"].fillna("").astype(str)
    panel_missing = df["panel_clean"].isna() | (panel_str.str.strip() == "")

    # Determine if each row is valid (blocking checks only)
    is_valid = (~id_missing) & (~des_missing) & (~too_short)

    # Now split the dataframe
    valid_df = df[is_valid].copy().reset_index(drop=True)
    rejected_df = df[~is_valid].copy()

    if not rejected_df.empty:
        rej_id_missing = id_missing.loc[~is_valid]
        rej_des_missing = des_missing.loc[~is_valid]
        rej_too_short = too_short.loc[~is_valid]
        rej_panel_missing = panel_missing.loc[~is_valid]
        rej_token_counts = token_counts.loc[~is_valid]

        reasons_zipped = zip(rej_id_missing, rej_des_missing, rej_too_short, rej_panel_missing, rej_token_counts)
        
        def build_reasons_str(id_m, des_m, short_m, panel_m, tokens):
            lst = []
            if id_m:
                lst.append("missing_id")
            if des_m:
                lst.append("null_designation")
            if short_m:
                lst.append(f"too_short ({tokens} tokens)")
            if panel_m:
                lst.append("null_panel")
            return ", ".join(lst)
            
        reasons_col = [build_reasons_str(*item) for item in reasons_zipped]
        rejected_df["rejected_reasons"] = reasons_col
    else:
        rejected_df["rejected_reasons"] = pd.Series(dtype=str)

    rejected_df = rejected_df.reset_index(drop=True)
    return valid_df, rejected_df

def validation_summary(
    total: int,
    valid_df: pd.DataFrame,
    rejected_df: pd.DataFrame
) -> dict:
    n_valid = len(valid_df)
    n_rejected = len(rejected_df)

    reason_counts = {}
    if not rejected_df.empty:
        for reasons in rejected_df["rejected_reasons"]:
            for r in reasons.split(", "):
                reason_counts[r] = reason_counts.get(r, 0) + 1
    
    return {
        "total":          total,
        "valid":          n_valid,
        "rejected":       n_rejected,
        "rejection_rate": round(n_rejected / total * 100, 2) if total else 0,
        "reasons":        reason_counts,
    }

def save_rejected(rejected_df: pd.DataFrame, run_id: str, chunk_name: str) -> None:
    """
    Appends rejected rows to the audit CSV (REJECTED_FILE).
    Adds run_id and chunk_name columns for traceability.
    Creates the file with a header on the first write.
    """
    if rejected_df.empty:
        return

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    out = rejected_df.copy()
    out.insert(0, "run_id", run_id)
    out.insert(1, "chunk", chunk_name)

    write_header = not REJECTED_FILE.exists()
    out.to_csv(
        REJECTED_FILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )

def run_deduplication(
    df: pd.DataFrame,
    existing_hashes: set[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Removes duplicate rows within the chunk (intra) and against
    already-seen hashes from previous chunks (cross).
    Hash key: id + designation_clean.
    Returns (deduped_df, stats).
    """
    df = df.copy()

    # Build a stable row hash from the two key columns
    df["row_hash"] = (
        df["id"].fillna("").astype(str)
        + "|"
        + df["designation_clean"].fillna("").astype(str)
    ).apply(lambda s: str(hash(s)))

    before = len(df)

    # Intra-chunk deduplication (keep first occurrence)
    df = df.drop_duplicates(subset=["row_hash"], keep="first")
    intra_dupes = before - len(df)

    # Cross-chunk deduplication (skip rows already in silver)
    cross_mask = df["row_hash"].isin(existing_hashes)
    cross_dupes = int(cross_mask.sum())
    df = df[~cross_mask].reset_index(drop=True)

    stats = {
        "intra_chunk_duplicates": intra_dupes,
        "cross_chunk_duplicates": cross_dupes,
    }
    return df, stats

def enrich_chunk(
    df: pd.DataFrame,
    source_file: str,
    run_id: str
) -> pd.DataFrame:
    """
    Adds traceability and derived columns required by silver/gold layers:
      - source_file     : originating bronze chunk filename
      - run_id          : pipeline run UUID
      - token_count     : word count of designation_clean
      - panel_canonical : rule-based canonical panel group from taxonomy
    """
    df = df.copy()
    df["source_file"] = source_file
    df["run_id"] = run_id
    df["token_count"] = (
        df["designation_clean"].fillna("").str.split().str.len()
    )
    df["panel_canonical"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[0] if isinstance(p, str) else None
    )
    df["category_l1"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[1] if isinstance(p, str) else None
    )
    df["category_l2"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[2] if isinstance(p, str) else None
    )
    df["category_l3"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[3] if isinstance(p, str) else None
    )
    df["category_l4"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[4] if isinstance(p, str) else None
    )
    df["category_l5"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[5] if isinstance(p, str) else None
    )
    return df

def process_chunk(
        chunk_path: Path,
        run_id: str,
        existing_hashes: set[str]
) -> tuple[pd.DataFrame, dict] | None:
    """
    Full pipeline for one bronze chunk.
    Returns (enriched_df, stats) or None if chunk was already processed.
    """
    chunk_name = chunk_path.name

    if is_chunk_processed(chunk_name):
        tqdm.write(f'    -> skip (already processed): {chunk_name}')
        return None
    
    t0 = time.time()
    df = pd.read_parquet(chunk_path)
    total_rows = len(df)

    df = normalize_chunk(df)
    valid_df, rejected_df = validate_chunk(df)
    val_summary = validation_summary(total_rows, valid_df, rejected_df)
    save_rejected(rejected_df, run_id, chunk_name)

    deduped_df, dedup_stats = run_deduplication(
        valid_df, existing_hashes
    )
    enriched_df = enrich_chunk(
        deduped_df, source_file=chunk_name, run_id=run_id
    )

    existing_hashes.update(enriched_df['row_hash'].tolist())

    mark_chunk_processed(chunk_name)

    duration = round(time.time() - t0, 2)

    stats = {
        "run_id": run_id,
        "chunk": chunk_name,
        "total_rows": total_rows,
        "valid_rows": val_summary["valid"],
        "rejected_rows": val_summary["rejected"],
        "rejection_rate": val_summary["rejection_rate"],
        "intra_duplicates": dedup_stats["intra_chunk_duplicates"],
        "cross_duplicates": dedup_stats["cross_chunk_duplicates"],
        "written_to_silver": len(enriched_df),
        "duration_s": duration
    }

    return enriched_df, stats

# SILVER
def silver_exists() -> bool:
    return SILVER_FILE.exists()

def read_silver() -> pd.DataFrame:
    if not silver_exists():
        return pd.DataFrame()
    return pd.read_parquet(SILVER_FILE)

def read_silver_hashes() -> pd.DataFrame:
    if not silver_exists():
        return set()
    df = pd.read_parquet(SILVER_FILE, columns=["row_hash"])
    return set(df['row_hash'].tolist())

def write_silver(df: pd.DataFrame) -> None:
    """
    Full overwrite - used on first run
    """
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER_FILE, index=False, compression='zstd')
    print(f'    -> silver written: {len(df):,}')

def append_silver(df: pd.DataFrame) -> None:
    """
    Append new rows to existing silver.
    Used on incremental runs.
    """
    if not silver_exists():
        write_silver(df)
        return
    
    existing = read_silver()
    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_parquet(SILVER_FILE, index=False, compression="zstd")
    print(f'    -> silver updated: {len(existing):,} -> {len(combined):,} rows (+{len(df):,})')

def silver_stats() -> dict:
    if not silver_exists():
        return {"exists": False}
    df = read_silver()
    return {
        "exists": True,
        "total_rows": len(df),
        "columns": df.columns.to_list(),
        "size_mb": round(SILVER_FILE.stat().st_size / 1024 / 1024, 2),
        "chunks_seen": df["source_file"].nunique() if "source_file" in df else 0,
        "runs_seen": df["run_id"].nunique() if "run_id" in df else 0
    }

def delete_rows_by_source(source_file: str) -> None:
    """
    Removes all rows from silver belonging to the given source file.
    """
    if not silver_exists():
        return
    df = read_silver()
    if "source_file" in df.columns:
        filtered_df = df[df["source_file"] != source_file]
        write_silver(filtered_df)
        print(f"    -> deleted rows for source {source_file}: {len(df) - len(filtered_df):,} rows removed")

# GOLD
# Columns kept from silver in the AI-ready gold layer.
# Must include every column referenced in build_gold().
GOLD_COLUMNS = [
    "id",
    "designation_clean",
    "panel_clean",
    "panel_canonical",
    "category_l1",
    "category_l2",
    "category_l3",
    "category_l4",
    "category_l5",
    "token_count",
    "row_hash",
    "source_file",
    "run_id",
]

def gold_exists() -> bool:
    return GOLD_FILE.exists()

def read_gold() -> pd.DataFrame: 
    if not GOLD_FILE.exists():
        raise FileNotFoundError("Gold file not found")
    return pd.read_parquet(GOLD_FILE)

def write_gold(df: pd.DataFrame) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GOLD_FILE, index=False, compression="zstd")
    print(f'    -> gold written: {len(df):,} rows -> {GOLD_FILE}')


def gold_stats() -> dict:
    if not gold_exists():
        return {"exists": False}
    df = read_gold()
    return {
        "exists": True,
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "size_mb": round(GOLD_FILE.stat().st_size / 1024 / 1024, 2),
        "avg_tokens": round(df["token_count"].mean(), 1),
        "max_tokens": int(df["token_count"].max()),
        "min_tokens": int(df["token_count"].min())
    }

def build_gold(silver_df: pd.DataFrame) -> pd.DataFrame:
    """
    TRANSFORMS SILVER INTO AI-READY GOLD:
    - Select only needed columns
    - Add any missing GOLD_COLUMNS (e.g. panel_canonical from old silver)
    - Drop rows with null designation_clean
    - Drop rows exceeding token limit
    - Drop rows with null panel_clean (optional but cleaner for AI)
    - Reset index
    """
    df = silver_df.copy()
    # Recompute panel_canonical and categories from latest taxonomy.json patterns
    print(f"  [gold] Computing panel_canonical and categories for {len(df):,} rows...")
    df["panel_canonical"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[0] if isinstance(p, str) else None
    )
    df["category_l1"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[1] if isinstance(p, str) else None
    )
    df["category_l2"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[2] if isinstance(p, str) else None
    )
    df["category_l3"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[3] if isinstance(p, str) else None
    )
    df["category_l4"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[4] if isinstance(p, str) else None
    )
    df["category_l5"] = df["panel_clean"].apply(
        lambda p: canonicalize_panel(p)[5] if isinstance(p, str) else None
    )
    cols = [c for c in GOLD_COLUMNS if c in df.columns]
    df = df[cols].copy()
    before = len(df)
    df = df[df["panel_clean"].notna()]
    df = df[df["token_count"] <= MAX_TOKENS]
    after = len(df)
    print(f"  [gold] Filtered: {before:,} → {after:,} rows ({before - after:,} dropped)")
    print(f"  [gold] Sorting...")
    df = df.sort_values(["panel_clean", "token_count"], ascending=[True, True])
    return df.reset_index(drop=True)

def run_gold_layer():
    """
    Reads silver, builds gold, writes it.
    """
    if not silver_exists():
        raise RuntimeError("Silver layer not found.")
    
    print(f"  [gold] Reading silver...")
    silver_df = read_silver()
    print(f"  [gold] Building gold from {len(silver_df):,} rows...")
    gold_df = build_gold(silver_df)
    print(f"  [gold] Writing {len(gold_df):,} rows to gold...")
    write_gold(gold_df)

# STATE/AUDIT
def load_state() -> dict:
    if not STATE_FILE.exists():
        return {
            "last_run": None,
            "files_processed": {},
            "chunks_processed": [],
            "total_rows": 0,
            "runs": 0
        }
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state: dict) -> None:
    """Persists the state dict to STATE_FILE as JSON."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

def compute_file_hash(path: Path) -> str:
    """Returns a stable MD5 hex digest of the file at `path`."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def mark_chunk_processed(chunk_name: str) -> None:
    """Records that a chunk has been successfully processed."""
    state = load_state()
    if chunk_name not in state["chunks_processed"]:
        state["chunks_processed"].append(chunk_name)
    save_state(state)

def is_chunk_processed(chunk_name: str) -> bool:
    """Returns True if the chunk has already been successfully processed."""
    state = load_state()
    return chunk_name in state.get("chunks_processed", [])

def mark_file_processed(raw_file: Path, total_written: int) -> None:
    """Records that a raw file has been processed and stores its hash."""
    state = load_state()
    state["files_processed"][raw_file.name] = compute_file_hash(raw_file)
    state["total_rows"] = state.get("total_rows", 0) + total_written
    state["runs"] = state.get("runs", 0) + 1
    state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_state(state)

def is_file_processed(raw_file: Path) -> bool:
    state = load_state()
    return raw_file.name in state.get("files_processed", {})

def reset_file_state(raw_file: Path) -> None:
    """Removes a file's entry from state so it gets reprocessed."""
    state = load_state()
    state["files_processed"].pop(raw_file.name, None)
    # Also clear chunks so they are reprocessed
    state["chunks_processed"] = []
    save_state(state)

def get_state_summary() -> dict:
    state = load_state()
    return {
        "last_run":         state.get("last_run"),
        "runs":             state.get("runs", 0),
        "total_rows":       state.get("total_rows", 0),
        "files_processed":  len(state.get("files_processed", {})),
        "chunks_processed": len(state.get("chunks_processed", [])),
    }

def log_run(stats: dict) -> None:
    import csv 
    from config.settings import RUN_LOG_FILE
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RUN_LOG_FILE.exists()
    with open(RUN_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(stats)

# ============================================================
# ROUTER
# ============================================================

# STRATEGIES
STRATEGY_FULL        = "FULL"         # silver is empty — process everything
STRATEGY_INCREMENTAL = "INCREMENTAL"  # new file — append only new rows
STRATEGY_UPSERT      = "UPSERT"       # existing file was modified — reprocess
STRATEGY_SKIP        = "SKIP"         # file already processed, unchanged

# DECISION LOGIC
def decide_strategy(raw_file: Path) -> str:
    """
    Decides which processing strategy to apply for a given raw file.
    """
    if not silver_exists():
        return STRATEGY_FULL
    
    state = load_state()
    name = raw_file.name
    stored = state['files_processed'].get(name)

    if stored is None:
        return STRATEGY_INCREMENTAL
    
    current = compute_file_hash(raw_file)
    if current == stored:
        return STRATEGY_SKIP
    
    return STRATEGY_UPSERT

def plan_pipeline(raw_files: list[Path]) -> list[tuple[Path, str]]:
    """
    Returns a list of (file, strategy) pairs for all raw files.
    """
    plan = [(f, decide_strategy(f)) for f in raw_files]
    for f, strategy in plan:
        print(f'    {f.name} -> {strategy}')
    print()
    return plan

# ============================================================
# PIPELINE
# ============================================================
def run_pipeline() -> None:
    run_id = str(uuid.uuid4())
    raw_files = get_raw_files()
    plan = plan_pipeline(raw_files)

    print(f'\n{'='*60}')
    print(f'    ETL PIPELINE - RUN ID: {run_id}')
    print(f'{'='*60}\n')

    existing_hashes = read_silver_hashes()

    for raw_file, strategy in plan:
        if strategy == STRATEGY_SKIP:
            print(f"    -> skip {raw_file.name} (already processed, unchanged)")
            continue

        if strategy == STRATEGY_UPSERT:
            print(f"UPSERT detected - {raw_file.name} was modified")
            
            # remove silver rows from this source
            chunks_to_delete = sorted(BRONZE_DIR.glob("chunk_*.parquet"))
            for c in chunks_to_delete:
                delete_rows_by_source(c.name)
                c.unlink()
                print(f"    deleted bronze chunk: {c.name}")
            
            # reset state so chunks are reprocessed
            reset_file_state(raw_file)
            existing_hashes = read_silver_hashes()
        
        chunk_raw_file(raw_file)

        chunks = sorted(BRONZE_DIR.glob("chunk_*.parquet"))

        processed_dfs = []
        total_written = 0
        for chunk_path in tqdm(chunks, desc='   Processing'):
            res = process_chunk(chunk_path, run_id, existing_hashes)
            if res is None: continue
            chunk_df, stats = res
            log_run(stats)
            processed_dfs.append(chunk_df)
            total_written += stats["written_to_silver"]
            tqdm.write(
                f"  -> {stats['chunk']} | "
                f"valid: {stats['valid_rows']:,} | "
                f"rejected: {stats['rejected_rows']:,} | "
                f"written: {stats['written_to_silver']:,} | "
                f"{stats['duration_s']}s"
            )
        
        if processed_dfs:
            combined_new = pd.concat(processed_dfs, ignore_index=True)
            append_silver(combined_new)
        
        mark_file_processed(raw_file, total_written)
    run_gold_layer()

    # QUALITY REPORT
    report = generate_report()
    print_report(report)
    save_report(report)

    # FINAL SUMMARY
    print(f"\n{'='*60}")

    print("SILVER STATS:")
    for k, v in silver_stats().items():
        print(f"    {k}: {v}")

    print("GOLD STATS:")
    for k, v in gold_stats().items():
        print(f"    {k}: {v}")

    print("\n STATE:")
    for k, v in get_state_summary().items():
        print(f"    {k}: {v}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    run_pipeline()