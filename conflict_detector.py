import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from segmenter import Clause
from models.schemas import ClauseType, RiskLevel, Conflict


# --- Rule-based conflict patterns ---
# Each rule: (type_a, type_b, check_function, description, severity)

def _extract_jurisdictions(text: str) -> list[str]:
    """Extract mentioned countries/cities from text."""
    locations = [
        "india", "uk", "united kingdom", "england", "usa", "united states",
        "singapore", "germany", "france", "dubai", "uae", "australia",
        "mumbai", "delhi", "london", "new york", "hong kong"
    ]
    return [loc for loc in locations if loc in text.lower()]


def check_jurisdiction_conflict(clause_a: Clause, clause_b: Clause) -> tuple[bool, str]:
    """Governing law vs dispute resolution — different jurisdictions."""
    locs_a = _extract_jurisdictions(clause_a.text)
    locs_b = _extract_jurisdictions(clause_b.text)
    overlap = set(locs_a) & set(locs_b)
    if locs_a and locs_b and not overlap:
        return True, f"Governing law references {locs_a} but dispute resolution references {locs_b} — jurisdiction mismatch."
    return False, ""


def check_termination_liability(clause_a: Clause, clause_b: Clause) -> tuple[bool, str]:
    """Immediate termination with uncapped liability is high risk."""
    has_immediate = any(kw in clause_a.text.lower() for kw in ["immediate termination", "immediately terminate"])
    has_uncapped = any(kw in clause_b.text.lower() for kw in ["unlimited liability", "uncapped", "no cap"])
    if has_immediate and has_uncapped:
        return True, "Immediate termination right combined with uncapped liability creates extreme financial exposure."
    return False, ""


def check_ip_profit_conflict(clause_a: Clause, clause_b: Clause) -> tuple[bool, str]:
    """One party owns all IP but gets minority profit share."""
    owns_all = any(kw in clause_a.text.lower() for kw in ["solely owned", "exclusively owned", "all ip belongs"])
    minority_share = any(kw in clause_b.text.lower() for kw in ["40%", "30%", "20%", "minority"])
    if owns_all and minority_share:
        return True, "One party holds all IP ownership but receives minority profit share — misaligned incentives."
    return False, ""


def check_confidentiality_survival(clause_a: Clause, clause_b: Clause) -> tuple[bool, str]:
    """Entire agreement clause may override confidentiality survival."""
    has_survival = "survive" in clause_a.text.lower()
    overrides_all = any(kw in clause_b.text.lower() for kw in ["supersedes all", "entire understanding", "whole agreement"])
    if not has_survival and overrides_all:
        return True, "Entire agreement clause may override confidentiality obligations — survival period not explicitly stated."
    return False, ""


def check_dual_governing_law(clause_a: Clause, clause_b: Clause) -> tuple[bool, str]:
    """Two governing law clauses with different jurisdictions."""
    locs_a = _extract_jurisdictions(clause_a.text)
    locs_b = _extract_jurisdictions(clause_b.text)
    if locs_a and locs_b and set(locs_a) != set(locs_b):
        return True, f"Multiple governing law clauses reference different jurisdictions: {locs_a} vs {locs_b}."
    return False, ""


# Maps conflict-prone clause type pairs to their check function and severity
CONFLICT_RULES = [
    (ClauseType.GOVERNING_LAW,    ClauseType.DISPUTE_RESOLUTION, check_jurisdiction_conflict,    RiskLevel.HIGH),
    (ClauseType.TERMINATION,      ClauseType.LIABILITY,          check_termination_liability,    RiskLevel.HIGH),
    (ClauseType.IP_OWNERSHIP,     ClauseType.PROFIT_SHARING,     check_ip_profit_conflict,       RiskLevel.MEDIUM),
    (ClauseType.CONFIDENTIALITY,  ClauseType.ENTIRE_AGREEMENT,   check_confidentiality_survival, RiskLevel.MEDIUM),
    (ClauseType.GOVERNING_LAW,    ClauseType.GOVERNING_LAW,      check_dual_governing_law,       RiskLevel.HIGH),
]


def run_rule_based_checks(clauses: list[Clause]) -> list[Conflict]:
    """Check all conflict-prone clause pairs using static rules."""
    conflicts = []
    clause_map = {}

    # Group clauses by type
    for clause in clauses:
        ctype = clause.clause_type
        if ctype not in clause_map:
            clause_map[ctype] = []
        clause_map[ctype].append(clause)

    for type_a, type_b, check_fn, severity in CONFLICT_RULES:
        clauses_a = clause_map.get(type_a, [])
        clauses_b = clause_map.get(type_b, [])

        # Same type conflict (e.g. two governing law clauses)
        if type_a == type_b:
            for i in range(len(clauses_a)):
                for j in range(i + 1, len(clauses_a)):
                    found, description = check_fn(clauses_a[i], clauses_a[j])
                    if found:
                        conflicts.append(Conflict(
                            clause_id_a=clauses_a[i].clause_id,
                            clause_id_b=clauses_a[j].clause_id,
                            conflict_description=description,
                            severity=severity
                        ))
        else:
            for ca in clauses_a:
                for cb in clauses_b:
                    found, description = check_fn(ca, cb)
                    if found:
                        conflicts.append(Conflict(
                            clause_id_a=ca.clause_id,
                            clause_id_b=cb.clause_id,
                            conflict_description=description,
                            severity=severity
                        ))

    return conflicts


# --- LLM conflict detection ---

LLM_CONFLICT_PROMPT = """You are a senior contract lawyer reviewing two clauses for contradictions.

Clause {id_a} ({type_a}):
{text_a}

Clause {id_b} ({type_b}):
{text_b}

Do these two clauses contradict or conflict with each other in any legally significant way?
Respond in this exact format:
CONFLICT: yes/no
SEVERITY: high/medium/low/none
DESCRIPTION: <one sentence explaining the conflict, or 'No conflict found'>"""


def run_llm_checks(clauses: list[Clause], client: Groq) -> list[Conflict]:
    """Use Groq to detect semantic conflicts across high-risk clause pairs."""
    conflicts = []

    # Only run LLM on these high-value pairs
    LLM_PAIRS = [
        (ClauseType.TERMINATION,     ClauseType.INDEMNITY),
        (ClauseType.LIABILITY,       ClauseType.INDEMNITY),
        (ClauseType.IP_OWNERSHIP,    ClauseType.CONFIDENTIALITY),
        (ClauseType.GOVERNING_LAW,   ClauseType.DISPUTE_RESOLUTION),
        (ClauseType.PROFIT_SHARING,  ClauseType.PAYMENT),
    ]

    clause_map = {}
    for clause in clauses:
        if clause.clause_type not in clause_map:
            clause_map[clause.clause_type] = []
        clause_map[clause.clause_type].append(clause)

    for type_a, type_b in LLM_PAIRS:
        for ca in clause_map.get(type_a, []):
            for cb in clause_map.get(type_b, []):
                prompt = LLM_CONFLICT_PROMPT.format(
                    id_a=ca.clause_id, type_a=type_a.value, text_a=ca.text[:400],
                    id_b=cb.clause_id, type_b=type_b.value, text_b=cb.text[:400]
                )
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0
                )
                raw = response.choices[0].message.content.strip()
                has_conflict = False
                severity = RiskLevel.NONE
                description = ""

                for line in raw.splitlines():
                    if line.startswith("CONFLICT:") and "yes" in line.lower():
                        has_conflict = True
                    elif line.startswith("SEVERITY:"):
                        val = line.split(":", 1)[1].strip().lower()
                        severity = RiskLevel(val) if val in RiskLevel._value2member_map_ else RiskLevel.NONE
                    elif line.startswith("DESCRIPTION:"):
                        description = line.split(":", 1)[1].strip()

                if has_conflict and description and description != "No conflict found":
                    conflicts.append(Conflict(
                        clause_id_a=ca.clause_id,
                        clause_id_b=cb.clause_id,
                        conflict_description=description,
                        severity=severity
                    ))

    return conflicts


# --- Main entry point ---

def detect_conflicts(clauses: list[Clause], groq_api_key: str = "") -> list[Conflict]:
    """Run rule-based + LLM conflict detection across all clauses."""
    conflicts = run_rule_based_checks(clauses)

    if groq_api_key or os.getenv("GROQ_API_KEY"):
        client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))
        llm_conflicts = run_llm_checks(clauses, client)
        conflicts.extend(llm_conflicts)

    return conflicts