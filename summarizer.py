import os
from groq import Groq
from dotenv import load_dotenv
from segmenter import Clause
from schemas import RiskScore, Conflict, NegotiationSuggestion, RiskLevel

load_dotenv()

BUSINESS_SUMMARY_PROMPT = """You are summarizing a contract for a business executive who is not a lawyer.

Contract Type: {doc_type}
Total Clauses: {total_clauses}

High Risk Clauses:
{high_risk}

Medium Risk Clauses:
{medium_risk}

Conflicts Found:
{conflicts}

Write a plain English business summary (3-4 sentences) covering:
1. What this contract is about
2. The biggest risks to watch out for
3. Any internal contradictions that need resolving
4. Overall recommendation (sign as-is / negotiate / do not sign)

Be direct and avoid legal jargon."""


LEGAL_SUMMARY_PROMPT = """You are a senior contract lawyer writing a legal review memo.

Contract Type: {doc_type}
Total Clauses: {total_clauses}

Risk Scores by Clause:
{risk_details}

Conflicts Detected:
{conflicts}

Negotiation Points:
{suggestions}

Write a concise legal summary (4-5 sentences) covering:
1. Overall risk profile of the contract
2. Most critical legal issues
3. Key negotiation points
4. Jurisdiction and enforceability concerns if any

Use precise legal language."""


def _format_risk_list(risk_scores: list[RiskScore], level: RiskLevel, clauses: list[Clause]) -> str:
    clause_map = {c.clause_id: c for c in clauses}
    items = [r for r in risk_scores if r.overall_risk == level]
    if not items:
        return "None"
    lines = []
    for r in items:
        clause = clause_map.get(r.clause_id)
        heading = clause.heading if clause else r.clause_id
        lines.append(f"- {heading}: {r.reasoning}")
    return "\n".join(lines)


def _format_conflicts(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "None"
    return "\n".join([f"- [{c.clause_id_a}] vs [{c.clause_id_b}] ({c.severity.value}): {c.conflict_description}" for c in conflicts])


def _format_risk_details(risk_scores: list[RiskScore], clauses: list[Clause]) -> str:
    clause_map = {c.clause_id: c for c in clauses}
    lines = []
    for r in risk_scores:
        clause = clause_map.get(r.clause_id)
        heading = clause.heading if clause else r.clause_id
        lines.append(f"- {heading}: Overall={r.overall_risk.value}, Legal={r.legal_risk.value}, Financial={r.financial_risk.value}, Operational={r.operational_risk.value}")
    return "\n".join(lines)


def _format_suggestions(suggestions: list[NegotiationSuggestion]) -> str:
    if not suggestions:
        return "None"
    return "\n".join([f"- [{s.clause_id}]: {s.rationale}" for s in suggestions])


def summarize(
    doc_type: str,
    clauses: list[Clause],
    risk_scores: list[RiskScore],
    conflicts: list[Conflict],
    suggestions: list[NegotiationSuggestion],
    groq_api_key: str = ""
) -> tuple[str, str]:
    """Returns (business_summary, legal_summary)."""

    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return (
            "No API key provided — business summary unavailable.",
            "No API key provided — legal summary unavailable."
        )

    client = Groq(api_key=api_key)

    # --- Business Summary ---
    business_prompt = BUSINESS_SUMMARY_PROMPT.format(
        doc_type=doc_type,
        total_clauses=len(clauses),
        high_risk=_format_risk_list(risk_scores, RiskLevel.HIGH, clauses),
        medium_risk=_format_risk_list(risk_scores, RiskLevel.MEDIUM, clauses),
        conflicts=_format_conflicts(conflicts)
    )

    business_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": business_prompt}],
        max_tokens=200,
        temperature=0.3
    )
    business_summary = business_response.choices[0].message.content.strip()

    # --- Legal Summary ---
    legal_prompt = LEGAL_SUMMARY_PROMPT.format(
        doc_type=doc_type,
        total_clauses=len(clauses),
        risk_details=_format_risk_details(risk_scores, clauses),
        conflicts=_format_conflicts(conflicts),
        suggestions=_format_suggestions(suggestions)
    )

    legal_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": legal_prompt}],
        max_tokens=250,
        temperature=0.2
    )
    legal_summary = legal_response.choices[0].message.content.strip()


    return business_summary, legal_summary
