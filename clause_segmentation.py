import re
import spacy
import json

nlp = spacy.load("en_core_web_sm")

SECTION_REGEX = re.compile(
    r"^(\d+(\.\d+)*|\([a-z]\)|[A-Z][A-Z\s]{5,})"
)

def preprocess_text(text: str) -> str:
    return text.replace("\t", " ").strip()

def extract_clauses(contract_text: str):
    clauses = []
    current_clause = None

    lines = preprocess_text(contract_text).split("\n")

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        if SECTION_REGEX.match(line):
            if current_clause:
                clauses.append(current_clause)

            current_clause = {
                "clause_id": f"CL_{len(clauses)+1:03}",
                "heading": line,
                "text": "",
                "start_line": idx
            }
        else:
            if current_clause:
                current_clause["text"] += " " + line

    if current_clause:
        clauses.append(current_clause)

    return clauses



CANONICAL_HEADINGS = {
    "liability": ["liability", "limitation", "damages"],
    "termination": ["termination", "terminate", "expiry"],
    "confidentiality": ["confidential", "nda"],
    "governing_law": ["law", "jurisdiction", "governing"],
    "payment": ["payment", "fees", "compensation"]
}

def normalize_heading(heading: str) -> str:
    heading_lower = heading.lower()
    for canonical, keywords in CANONICAL_HEADINGS.items():
        if any(keyword in heading_lower for keyword in keywords):
            return canonical
    return "other"

