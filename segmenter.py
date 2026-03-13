import re
from dataclasses import dataclass, field
from typing import Optional
from models.schemas import ClauseType


# --- Output ---

@dataclass
class Clause:
    clause_id: str
    heading: Optional[str]
    text: str
    start_line: int
    end_line: int
    confidence: float   # 1.0 = clean numbered heading, 0.6 = all-caps, 0.4 = fallback
    clause_type: Optional[ClauseType] = ClauseType.UNKNOWN
    deviation_score: Optional[float] = None      # 0.0 = matches standard, 1.0 = highly deviant
    top_reference: Optional[str] = None          # closest matching reference clause
    reference_similarity: Optional[float] = None # raw cosine similarity
    match_quality: Optional[str] = None          # good / weak / none


# --- Heading patterns (ordered by confidence) ---

NUMBERED_HEADING = re.compile(
    r'^\s*(\d+(\.\d+)*\.?)\s+([A-Z][A-Za-z\s\-/&]{2,60})\s*$'
)  # e.g. "1. Confidentiality" or "3.1 Payment Terms"

CAPS_HEADING = re.compile(
    r'^\s*([A-Z][A-Z\s\-/&]{4,60})\s*$'
)  # e.g. "GOVERNING LAW" or "FORCE MAJEURE"

CLAUSE_LABEL = re.compile(
    r'^\s*(CLAUSE|SECTION|ARTICLE)\s+\d+[\.:]\s*(.+)$',
    re.IGNORECASE
)  # e.g. "CLAUSE 5: Termination"


def detect_heading(line: str) -> tuple[Optional[str], float]:
    """Returns (heading_text, confidence) or (None, 0.0)"""

    m = CLAUSE_LABEL.match(line)
    if m:
        return m.group(0).strip(), 1.0

    m = NUMBERED_HEADING.match(line)
    if m:
        return line.strip(), 1.0

    m = CAPS_HEADING.match(line)
    if m:
        return line.strip(), 0.6

    return None, 0.0


# --- Segmenter ---

def segment(raw_text: str) -> list[Clause]:
    lines = raw_text.splitlines()
    clauses = []
    current_heading = None
    current_confidence = 0.4
    current_start = 0
    current_lines = []
    clause_index = 0

    def save_clause():
        nonlocal clause_index
        text = "\n".join(current_lines).strip()
        if not text:
            return
        clause_index += 1
        clauses.append(Clause(
            clause_id=f"CL-{clause_index:03d}",
            heading=current_heading,
            text=text,
            start_line=current_start,
            end_line=current_start + len(current_lines) - 1,
            confidence=current_confidence
        ))

    for i, line in enumerate(lines):
        heading, confidence = detect_heading(line)

        if heading:
            # Save whatever we've accumulated so far
            save_clause()
            # Start new clause
            current_heading = heading
            current_confidence = confidence
            current_start = i
            current_lines = []
        else:
            current_lines.append(line)

    # Save the last clause
    save_clause()

    return clauses