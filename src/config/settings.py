import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() #loads .env into os.environ silently if file exists

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
CLEAN_DIR = DATA_DIR / "clean"
AUDIT_DIR = DATA_DIR / "audit"
STATE_DIR = DATA_DIR / "state"
TAXONOMY_DIR = DATA_DIR / "taxonomy"

RAW_FILE = RAW_DIR / "Article.csv.gz"
SILVER_FILE = SILVER_DIR / "Article_clean.parquet"
GOLD_FILE = GOLD_DIR / "Article_ai_ready.parquet"
STATE_FILE = STATE_DIR / "etl_state.json"
RUN_LOG_FILE = AUDIT_DIR / "run_log.csv"
REJECTED_FILE = AUDIT_DIR / "rejected_rows.csv"

TAXONOMY_FILE = PROJECT_ROOT / "src" / "config" / "taxonomy_5level.json"
CLASSIFIER_CACHE = TAXONOMY_DIR / "classifier_cache.json"
TAXONOMY_MAP_FILE = TAXONOMY_DIR / "taxonomy_map.parquet"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_VALIDATION_MODEL = os.getenv("OLLAMA_VALIDATION_MODEL", "qwen2.5:3b-instruct")

PANEL_MIN_FREQ_FOR_CLASSIFY = int(os.getenv("PANEL_MIN_FREQ_FOR_CLASSIFY", "5"))
VALIDATION_CONFIDENCE_THRESHOLD = float(os.getenv("VALIDATION_CONFIDENCE_THRESHOLD", "0.85"))

CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"
REQUIRED_COLS = ['id', 'panel', 'designation']
CHUNK_SIZE = 100_000
MAX_TOKENS = 300

BOILERPLATE_PATTERNS = {
    # 1. Le bloc Éditions/Librairie (Motif massif repéré de la ligne 91 à 189)
    'has_clause_editeur': r'(?i)(edition|dar|adar)\s+[a-z\s0-9\(\)]+?\s+ou\s*autres?',
    
    # 2. Logistique, Matériel et Service (Repéré dans l'événementiel/restauration ex: 39, 40, 89)
    'has_service_materiel': r'(?i)(mat[eé]riel\s+et\s+service\s+complet|y/c\s+service\s+et\s+mat[eé]riel)',
    
    # 3. Clauses Administratives et Maître d'Ouvrage (ex: 20, 38, 45, 69)
    'has_clause_mo': r'(?i)(demand[eé]\s+par\s+M\.\s*O|selon\s+ma[iî]tre\s+d[\'’]\s*ouvrage)',
    'has_obligation_justificatif': r'(?i)la\s+pr[eé]sentation\s+du\s+re[cç]u\s+de\s+confirmation.*obligatoire',
    
    # 4. Restriction Territoriale et Sous-traitance (Crucial pour le prix, ex: 14, 37)
    'has_interdit_soustraitance': r'(?i)ne\s+peux?\s+en\s+aucun\s+cas\s+sous-traiter.*',
    'has_zone_geographique': r'(?i)ex[eé]cut[eé]s?\s+dans\s+le\s+territoire\s+de\s+la\s+pr[eé]fecture.*',
    
    # 5. Métadonnées d'exécution (Injectées par des outils d'extraction ou de saisie, ex: 42, 43, 45)
    'meta_lieu_date_execution': r'(?i)(lieu|date)\s+d[\'’]?[eé]x[eé]cution\s*:\s*[^;\n\.]+',

    # 6. Reprise et robustification des basiques (Garantie, Sujétions, Fourniture)
    'has_garantie': r'(?i)(\d+\s*ann[eé]es?\s+de\s+)?garantie\s*[:\-]?\s*\d+\s*ans?',
    'has_sujetions': r'(?i)toutes?\s+suj[eé]tions?(\s+de\s+(fourniture|pose|mise\s+en\s+œuvre|raccordement|fournitures))?(\s+incluses?)?',
    'has_fourniture_pose': r'(?i)(y\s+compris\s+|fourni\s*,\s*p[oî]s[eé]\s+et\s+en\s+)?fournitures?\s+et\s+pose(\s+de)?',
    'has_regles_art': r'(?i)selon\s+les\s+r[eé]gles?\s+de\s+l[\'’]\s*art',
    'has_similaire': r'(?i)ou\s+similaire|ou\s+équivalent',
    'has_qualite_premium': r'(?i)1[eé]r\s+choix|1er\s+choix|meilleure\s+qualit[eé]|haute\s+performance|qualit[eé]\s+pro|tr[eé]s\s+bon\s+[eé]tat'
}
NULL_MARKERS = ["-", "nan", "none", "null", "n/a", ""]
