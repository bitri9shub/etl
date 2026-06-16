import json
import time
import requests
from pathlib import Path
from itertools import islice

from config.settings import TAXONOMY_FILE, CLASSIFIER_CACHE, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_VALIDATION_MODEL, OLLAMA_TIMEOUT, VALIDATION_CONFIDENCE_THRESHOLD

CLASSIFY_PROMPT = """
## ROLE
You are a specialized procurement taxonomy classifier operating on Moroccan public procurement data (marchés publics).
Your task: map a raw panel name to the single most precise node in a 5-level taxonomy tree.

## INPUT CONTEXT
- Panel names are raw, noisy labels extracted from purchase orders (bons de commande)
- Language: French, Arabic, Darija, English, or mixed — classify by SEMANTIC MEANING, not surface form
- Common noise patterns: typos ("tonner" → toner), abbreviations ("frntr" → mobilier), acronyms ("EPI" → Équipements de Protection Individuelle), partial phrases
- Domain: Moroccan public procurement — expect administrative supplies, civil works, IT equipment, medical devices, catering, fuel, services

## TAXONOMY TREE
{tree}

## CLASSIFICATION PROTOCOL
Execute the following reasoning chain internally before producing output:

1. NORMALIZE  — strip noise, fix typos, expand abbreviations, identify core product/service concept
2. TRANSLATE  — if Arabic/Darija, resolve semantic meaning (do not transliterate blindly)
3. ANCHOR     — identify the best-matching L1 category; if ambiguous between two, prefer the one with a more specific L3/L4/L5 match
4. DESCEND    — traverse L2 → L3 → L4 → L5, stopping at the deepest node with confident evidence
5. CALIBRATE  — assign confidence:
   • 0.90-1.00 : exact or near-exact match, unambiguous product/service
   • 0.70-0.89 : strong semantic match, minor ambiguity (e.g. brand name without category context)
   • 0.50-0.69 : plausible match but panel name is too vague or multi-category
   • 0.00-0.49 : weak match — prefer "Non classé" below 0.40
6. FALLBACK   — if no branch has confidence ≥ 0.40, classify as "Non classé" at all levels

## GROUPING RULES
Group semantically equivalent variants under the same canonical label:
  - "toner", "tonner", "cartouche d'encre", "ink cartridge"  → canonical: "Toner / Cartouche"
  - "chaise de bureau", "siège bureau", "office chair"        → canonical: "Siège de Bureau"
  - "travaux peinture", "peinture bâtiment", "طلاء"          → canonical: "Travaux de Peinture"
Apply this logic to the canonical field — normalize to a clean, French procurement label.

## PANEL TO CLASSIFY
"{panel}"

## OUTPUT FORMAT
Respond with ONLY a single valid JSON object. No explanation, no markdown, no extra keys.

{{
  "canonical": "<normalized French procurement label, or empty string if Non classé>",
  "l1": "<L1 category or Non classé>",
  "l2": "<L2 category or Non classé>",
  "l3": "<L3 category or Non classé>",
  "l4": "<L4 category or Non classé>",
  "l5": "<L5 category or Non classé>",
  "confidence": <float 0.0-1.0>,
  "match_rationale": "<one sentence: what concept was matched and why, or why fallback was triggered>"
}}
"""

VALIDATION_PROMPT = """
## ROLE
You are an adversarial taxonomy auditor for Moroccan public procurement (marchés publics).
Your job is NOT to rubber-stamp the classifier's output — it is to actively challenge it.
Assume the classifier made an error until the evidence proves otherwise.

## CONTEXT
- Source data: raw panel names from bons de commande (purchase orders), noisy, multilingual
- Classifier confidence provided — treat low confidence as a strong signal of likely error
- Taxonomy is 5-level hierarchical: L1 (broadest) → L5 (most specific)
- A valid path must be strictly coherent: each level must be a logical sub-category of the one above

## ALGORITHMIC CLASSIFICATION UNDER REVIEW
Panel (raw input) : "{panel}"
Canonical label  : "{canonical}"
L1               : "{l1}"
L2               : "{l2}"
L3               : "{l3}"
L4               : "{l4}"
L5               : "{l5}"
Classifier confidence : {confidence}
Classifier rationale  : "{match_rationale}"

## AVAILABLE TAXONOMY TREE
{tree}

## VALIDATION PROTOCOL
Execute ALL checks. Fail fast — one HARD FAIL is sufficient to trigger correction.

### CHECK 1 — PATH COHERENCE
Verify L1 → L2 → L3 → L4 → L5 form a valid descending path in the taxonomy tree.
A level containing "Non classé" must propagate to all deeper levels (no mix of real + Non classé).
❌ HARD FAIL if any level names a node that does not exist in the tree.
❌ HARD FAIL if a deeper level is broader than a shallower level.

### CHECK 2 — PANEL-CANONICAL ALIGNMENT
Does the canonical label faithfully represent the semantic core of the raw panel?
Account for: typos, abbreviations, Arabic/Darija terms, mixed-language noise.
❌ HARD FAIL if canonical describes a different product/service than the panel.
⚠ SOFT WARN if canonical is plausible but imprecise (e.g. too generic for the panel).

### CHECK 3 — SPECIFICITY CALIBRATION
Is the deepest populated level appropriate for the panel's detail level?
- A specific panel (e.g. "Cartouche HP LaserJet 85A") demands L4-L5 classification.
- A vague panel (e.g. "Fournitures diverses") may legitimately stop at L2-L3.
❌ HARD FAIL if a specific panel is classified at L1-L2 only (under-classification).
❌ HARD FAIL if a vague panel forces a specific L5 node (over-classification / hallucination).

### CHECK 4 — DOMAIN PLAUSIBILITY
Does this classification make sense in Moroccan public procurement RFQ context?
Known high-risk misclassification patterns:
  - Fuel ("gasoil", "carburant", "mazout") → must NOT land in Informatique or Chimie
  - Medical devices ("tensiomètre", "seringue") → must NOT land in Fournitures de Bureau
  - Civil works ("béton", "ferraillage", "enduit") → must NOT land in Mobilier
  - Catering ("plateau repas", "eau minérale") → must NOT land in Produits Chimiques
❌ HARD FAIL on any cross-domain collision.

### CHECK 5 — CONFIDENCE CONSISTENCY
Is the classifier's stated confidence consistent with the classification quality?
  - confidence ≥ 0.90 + HARD FAIL on any check → confidence was inflated, flag as OVERCONFIDENT
  - confidence < 0.50 + all checks pass → may be appropriate, flag as CONSERVATIVE
  - confidence < 0.40 + non-"Non classé" classification → flag as INCONSISTENT (should have fallen back)

## CORRECTION RULES (apply only if validation fails)
1. Re-classify from scratch using your own judgment over the taxonomy tree.
2. Do NOT preserve any field from the original classification if a HARD FAIL was triggered.
3. If the panel is genuinely unclassifiable (too vague, corrupted, non-procurement noise), set all levels to "Non classé", canonical = "", corrected_confidence = 0.0.
4. Provide a corrected_confidence reflecting your own certainty in the correction.

## OUTPUT FORMAT
Respond with ONLY a single valid JSON object. No explanation, no markdown, no extra keys.

{{
  "valid": <true | false>,
  "verdict": "<PASS | SOFT_WARN | HARD_FAIL>",
  "failed_checks": ["CHECK_N — reason", ...],
  "confidence_flag": "<OVERCONFIDENT | CONSERVATIVE | INCONSISTENT | OK | null>",
  "audit_note": "<one to two sentences summarizing the audit decision>",
  "corrected": {{
    "canonical": "...",
    "l1": "...",
    "l2": "...",
    "l3": "...",
    "l4": "...",
    "l5": "...",
    "corrected_confidence": <float 0.0-1.0>
  }} | null
}}

Rules for the top-level fields:
- "valid": true only if verdict is PASS (zero hard fails, zero soft warns)
- "failed_checks": empty list [] on PASS
- "corrected": non-null only when valid = false; must be null on PASS
"""

BOOTSTRAP_PROMPT = """
## ROLE
You are a senior procurement taxonomy architect specializing in Moroccan public procurement (marchés publics).
Your task: induce a principled 5-level taxonomy from a raw sample of panel names extracted from bons de commande.
This taxonomy will become the reference tree used by downstream LLM classifiers and validators — design it for machine readability and human auditability.

## MISSION CONSTRAINTS
- Source data is noisy: typos, abbreviations, Arabic/Darija terms, mixed-language labels, partial descriptions
- Domain is heterogeneous: a single dataset may contain PEHD pipes, catering, medical devices, fuel, IT equipment, civil works — all unlabeled
- The taxonomy must be OPEN-WORLD: favor generality at L1–L2, reserve specificity for L3–L5
- You are NOT asked to classify each panel — you are asked to INDUCE the category space that can cover them
- No hardcoded categories: derive structure from the data, guided by Moroccan procurement norms (Code des Marchés Publics)

## INPUT — RAW PANEL SAMPLE
{panel_sample}

## TAXONOMY DESIGN PROTOCOL

### PHASE 1 — SIGNAL EXTRACTION
Scan the panel list. For each entry:
- Normalize: strip noise, fix typos, expand abbreviations
- Identify core product/service concept (ignore quantity, brand, model number)
- Tag language: FR / AR / DA / EN / MIXED
- Note semantic cluster affinity (even rough grouping at this stage)

### PHASE 2 — L1 INDUCTION
Derive 8–15 top-level categories (L1) that are:
- Mutually exclusive: no panel should belong to two L1 nodes with equal plausibility
- Collectively exhaustive: the full Moroccan public procurement space should be coverable
- Aligned with standard procurement frameworks (UNSPSC, NACRES, CPV if applicable)
- Named in FRENCH, in noun-phrase form (e.g. "Fournitures de Bureau", not "Bureautique" alone)
Always include:
  • "Non classé" as a reserved L1 node for unresolvable panels
  • "Services" as a distinct L1 from "Fournitures" (goods vs. services separation is legally significant in marchés publics)

### PHASE 3 — L2–L5 DESCENT
For each L1 node:
- L2: major sub-domains (4–10 per L1)
- L3: product/service families (3–8 per L2)
- L4: specific product types (2–6 per L3, only where data supports it)
- L5: SKU-level or spec-level distinctions (only where panels are specific enough; leave as "—" otherwise)
Design rules:
  ✓ Each child node must be more specific than its parent
  ✓ Siblings at the same level must be disjoint
  ✓ Do not create a level-N node unless at least 2 panels in the sample support it
  ✓ Prefer 3-word max node names; use "/" for semantic alternatives ("Toner / Cartouche")

### PHASE 4 — MULTILINGUAL ALIASING
For each leaf node (deepest populated level), provide:
- Primary label: French
- Arabic alias (if relevant panels exist in Arabic)
- Common abbreviation or acronym if widely used (e.g. "EPI", "VRD", "GTC")

### PHASE 5 — COHERENCE REVIEW
Before finalizing:
- Verify no panel fits two L1 nodes equally — if so, refine L1 boundaries
- Verify "Non classé" is not used for panels that are clearly classifiable
- Verify depth is data-driven: do not hallucinate L4/L5 nodes not evidenced in the sample
- Flag under-represented L1 nodes (< 3 panels) as SPARSE — they may need merging

## OUTPUT FORMAT
Respond with ONLY a single valid JSON object. No explanation, no markdown, no extra keys.
Structure:

{{
  "taxonomy_version": "1.0",
  "induced_from_sample_size": <int>,
  "generated_at": "{timestamp}",
  "coverage_notes": "<1–2 sentences on what domains are well-represented vs. sparse in this sample>",
  "sparse_nodes": ["L1 node names flagged as under-represented"],
  "tree": [
    {{
      "l1": "Fournitures de Bureau",
      "l1_aliases": {{"ar": "لوازم مكتبية", "abbr": null}},
      "children": [
        {{
          "l2": "Papeterie",
          "children": [
            {{
              "l3": "Papier",
              "children": [
                {{
                  "l4": "Papier Reprographie",
                  "children": [
                    {{
                      "l5": "Papier A4 80g/m²",
                      "l5_aliases": {{"ar": null, "abbr": null}}
                    }}
                  ]
                }}
              ]
            }}
          ]
        }}
      ]
    }},
    {{
      "l1": "Non classé",
      "l1_aliases": {{"ar": "غير مصنف", "abbr": null}},
      "children": []
    }}
  ]
}}

## QUALITY GATES
Before emitting output, verify:
  □ tree contains between 8 and 15 L1 nodes (inclusive of "Non classé")
  □ "Non classé" node exists with empty children []
  □ "Services" exists as a distinct L1 node
  □ No L4/L5 node created without panel evidence
  □ All node names are in French noun-phrase form
  □ No node name exceeds 6 words
  □ sparse_nodes list is populated honestly (do not omit to appear comprehensive)
"""


class TaxonomyClassifier:
    def __init__(self, taxonomy_path: str | Path = TAXONOMY_FILE,
                 cache_path: str | Path = CLASSIFIER_CACHE,
                 base_url: str = OLLAMA_BASE_URL,
                 model: str = OLLAMA_MODEL,
                 validation_model: str = OLLAMA_VALIDATION_MODEL,
                 timeout: int = OLLAMA_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.validation_model = validation_model
        self.timeout = timeout
        self.taxonomy_path = Path(taxonomy_path)
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()
        self._tree_str = None

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _format_tree(self) -> str:
        if self._tree_str is not None:
            return self._tree_str
        if not self.taxonomy_path.exists():
            self._tree_str = "No taxonomy defined"
            return self._tree_str
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            tax = json.load(f)
        lines = []
        for l1, info in tax.get("tree", {}).items():
            children = info.get("children", [])
            for l2 in children:
                if isinstance(l2, dict):
                    l2_name = l2.get("name", list(l2.keys())[0] if l2 else "?")
                else:
                    l2_name = l2
                lines.append(f"  {l1} > {l2_name}")
        self._tree_str = "\n".join(lines)
        return self._tree_str

    def _ollama_generate(self, prompt: str, system: str | None = None) -> str | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 512, "temperature": 0.1}
        }
        if system:
            payload["system"] = system
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.RequestException as e:
            return None

    def classify(self, panel: str) -> dict:
        cached = self.cache.get(panel)
        if cached is not None:
            return cached

        tree = self._format_tree()
        prompt = CLASSIFY_PROMPT.format(tree=tree, panel=panel)
        raw = self._ollama_generate(prompt)

        result = self._parse_classification(raw, panel)

        if result.get("confidence", 0.0) >= VALIDATION_CONFIDENCE_THRESHOLD:
            validation_result = self._validate_classification(panel, result)
            if validation_result["valid"]:
                self.cache[panel] = validation_result
                self._save_cache()
                return validation_result

        self.cache[panel] = result
        self._save_cache()
        return result

    def classify_batch(self, panels: list[str]) -> dict[str, dict]:
        results = {}
        uncached = []
        for p in panels:
            cached = self.cache.get(p)
            if cached is not None:
                results[p] = cached
            else:
                uncached.append(p)

        for panel in uncached:
            result = self.classify(panel)
            results[panel] = result

        return results

    def _parse_classification(self, raw: str | None, panel: str) -> dict:
        if raw is None:
            return {"canonical": "", "l1": "Non classé", "l2": "Non classé", "l3": "Non classé", "l4": "Non classé", "l5": "Non classé", "confidence": 0.0}
        try:
            parsed = json.loads(raw)
            if all(k in parsed for k in ("canonical", "l1", "l2", "l3", "l4", "l5")):
                parsed.setdefault("confidence", 1.0)
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"canonical": "", "l1": "Non classé", "l2": "Non classé", "l3": "Non classé", "l4": "Non classé", "l5": "Non classé", "confidence": 0.0}

    def _validate_classification(self, panel: str, classification: dict) -> dict:
        confidence = classification.get("confidence", 0.0)
        if confidence < VALIDATION_CONFIDENCE_THRESHOLD:
            return classification

        validation_prompt = VALIDATION_PROMPT.format(
            panel=panel,
            canonical=str(classification.get("canonical", "")),
            l1=str(classification.get("l1", "Non classé")),
            l2=str(classification.get("l2", "Non classé")),
            l3=str(classification.get("l3", "Non classé")),
            l4=str(classification.get("l4", "Non classé")),
            l5=str(classification.get("l5", "Non classé"))
        )

        raw = self._ollama_generate(validation_prompt, system="You are a taxonomy validation expert. Validate classifications against RFQ procurement standards.")
        validation_result = self._parse_validation(raw)

        if validation_result["valid"] and validation_result.get("corrected"):
            return validation_result["corrected"]

        return classification

    def _parse_validation(self, raw: str | None) -> dict:
        if raw is None:
            return {"valid": False, "reason": "Validation failed", "corrected": None}
        try:
            parsed = json.loads(raw)
            if all(k in parsed for k in ("valid", "reason")):
                parsed.setdefault("corrected", None)
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return {"valid": False, "reason": "Validation failed", "corrected": None}

    def bootstrap_from_panels(self, panels: list[str], batch_size: int = 50) -> list[dict]:
        groups = []
        for i in range(0, len(panels), batch_size):
            batch = panels[i:i + batch_size]
            panel_text = "\n".join(f"- {p}" for p in batch)
            prompt = BOOTSTRAP_PROMPT.format(panels=panel_text)
            raw = self._ollama_generate(prompt, system="You are a taxonomy expert.")
            parsed = self._parse_bootstrap(raw)
            groups.extend(parsed)
        return groups

    def _parse_bootstrap(self, raw: str | None) -> list[dict]:
        if raw is None:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return []
