import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from segmenter import Clause
from schemas import ClauseType, RiskLevel, RiskScore


# --- Static risk rules ---

# Each rule is (pattern_keywords, risk_axis, risk_level, reason)
STATIC_RULES = [
    # Financial
    (["unlimited liability", "unlimited damages"],                          "financial", RiskLevel.HIGH,   "Unlimited liability exposure"),
    (["uncapped liability", "no cap on liability"],                         "financial", RiskLevel.HIGH,   "No liability cap defined"),
    (["auto-renewal", "automatically renew", "automatically extends"],      "financial", RiskLevel.MEDIUM, "Auto-renewal clause — easy to miss"),
    (["penalty", "liquidated damages"],                                     "financial", RiskLevel.MEDIUM, "Financial penalty clause present"),

    # Legal
    (["sole discretion", "absolute discretion"],                            "legal",     RiskLevel.HIGH,   "One-sided discretionary power"),
    (["irrevocable", "perpetual license"],                                  "legal",     RiskLevel.HIGH,   "Irrevocable or perpetual rights granted"),
    (["waive", "waiver of rights"],                                         "legal",     RiskLevel.MEDIUM, "Rights waiver present"),
    (["indemnify", "hold harmless"],                                        "legal",     RiskLevel.MEDIUM, "Indemnity obligations present"),
    (["exclusive jurisdiction"],                                            "legal",     RiskLevel.MEDIUM, "Exclusive jurisdiction clause — check location"),

    # Operational
    (["immediately terminate", "immediate termination"],                    "operational", RiskLevel.HIGH,   "Immediate termination right without cure period"),
    (["unreasonable", "best efforts", "commercially reasonable efforts"],   "operational", RiskLevel.MEDIUM, "Vague obligation standard"),
    (["30 days", "60 days", "90 days"],                                     "operational", RiskLevel.LOW,    "Notice period defined — verify if acceptable"),
    (["force majeure"],                                                     "operational", RiskLevel.LOW,    "Force majeure — check scope of covered events"),
]

RISK_ORDER = {RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}

def escalate(current: RiskLevel, new: RiskLevel) -> RiskLevel:
    """Return the higher of two risk levels."""
    return new if RISK_ORDER[new] > RISK_ORDER[current] else current


def apply_static_rules(clause: Clause) -> tuple[RiskLevel, RiskLevel, RiskLevel, list[str]]:
    text_lower = clause.text.lower()
    legal = RiskLevel.NONE
    financial = RiskLevel.NONE
    operational = RiskLevel.NONE
    reasons = []

    for keywords, axis, level, reason in STATIC_RULES:
        if any(kw in text_lower for kw in keywords):
            if axis == "legal":
                legal = escalate(legal, level)
            elif axis == "financial":
                financial = escalate(financial, level)
            elif axis == "operational":
                operational = escalate(operational, level)
            reasons.append(reason)

    return legal, financial, operational, reasons


# --- LLM risk scoring ---

LLM_PROMPT = """You are a senior legal risk analyst reviewing a contract clause.

Clause Type: {clause_type}
Clause Text: {clause_text}

Deviation from standard templates: {deviation_score} (0.0 = standard, 1.0 = highly unusual)
Static rule flags: {static_flags}

Assess the risk of this clause across three axes and provide a brief reasoning.
Respond in this exact format:
LEGAL: high/medium/low/none
FINANCIAL: high/medium/low/none
OPERATIONAL: high/medium/low/none
REASONING: <one or two sentences>"""


def apply_llm_scoring(clause: Clause, client: Groq, static_reasons: list[str]) -> tuple[RiskLevel, RiskLevel, RiskLevel, str]:
    prompt = LLM_PROMPT.format(
        clause_type=clause.clause_type.value if clause.clause_type else "unknown",
        clause_text=clause.text[:600],
        deviation_score=round(clause.deviation_score or 0.0, 2),
        static_flags=", ".join(static_reasons) if static_reasons else "None"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    legal = financial = operational = RiskLevel.NONE
    reasoning = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("LEGAL:"):
            legal = RiskLevel(line.split(":", 1)[1].strip().lower()) if line.split(":", 1)[1].strip().lower() in RiskLevel._value2member_map_ else RiskLevel.NONE
        elif line.startswith("FINANCIAL:"):
            financial = RiskLevel(line.split(":", 1)[1].strip().lower()) if line.split(":", 1)[1].strip().lower() in RiskLevel._value2member_map_ else RiskLevel.NONE
        elif line.startswith("OPERATIONAL:"):
            operational = RiskLevel(line.split(":", 1)[1].strip().lower()) if line.split(":", 1)[1].strip().lower() in RiskLevel._value2member_map_ else RiskLevel.NONE
        elif line.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return legal, financial, operational, reasoning


def compute_overall(legal: RiskLevel, financial: RiskLevel, operational: RiskLevel) -> RiskLevel:
    return max([legal, financial, operational], key=lambda r: RISK_ORDER[r])


# --- Main entry point ---

def score_risks(clauses: list[Clause], groq_api_key: str = "") -> list[RiskScore]:
    client = None
    if groq_api_key or os.getenv("GROQ_API_KEY"):
        client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))

    results = []

    for clause in clauses:
        # Step A: static rules
        s_legal, s_financial, s_operational, reasons = apply_static_rules(clause)

    # Step B: match quality signal — no reference match = escalate risk
        match_quality = getattr(clause, "match_quality", "good")
        if match_quality == "none":
            s_legal = escalate(s_legal, RiskLevel.MEDIUM)
            reasons.append("Clause has no close reference match — highly unusual language")
        elif match_quality == "weak":
            reasons.append("Clause weakly matches standard templates — review carefully")

        if client:
            # Step C: LLM scoring — escalate if LLM sees higher risk
            l_legal, l_financial, l_operational, reasoning = apply_llm_scoring(clause, client, reasons)
            final_legal = escalate(s_legal, l_legal)
            final_financial = escalate(s_financial, l_financial)
            final_operational = escalate(s_operational, l_operational)
        else:
            final_legal, final_financial, final_operational = s_legal, s_financial, s_operational
            reasoning = "; ".join(reasons) if reasons else "No flags from static rules."

        overall = compute_overall(final_legal, final_financial, final_operational)

        results.append(RiskScore(
            clause_id=clause.clause_id,
            legal_risk=final_legal,
            financial_risk=final_financial,
            operational_risk=final_operational,
            overall_risk=overall,
            reasoning=reasoning,
            deviation_score=clause.deviation_score
        ))


    return results
