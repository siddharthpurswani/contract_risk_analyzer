import os
from groq import Groq
from dotenv import load_dotenv
from segmenter import Clause
from schemas import RiskLevel, RiskScore, NegotiationSuggestion

load_dotenv()

# Only suggest changes for these risk levels
SUGGEST_FOR = {RiskLevel.HIGH, RiskLevel.MEDIUM}

NEGOTIATION_PROMPT = """You are an expert contract lawyer representing a client in a commercial partnership agreement.

The following clause has been flagged as {risk_level} risk for these reasons:
{reasoning}

Original clause ({clause_type}):
{clause_text}

Rewrite this clause to be more balanced and fair to both parties while preserving its intent.
Be specific — use concrete time periods, caps, and conditions where appropriate.

Respond in this exact format:
SUGGESTED: <rewritten clause text>
RATIONALE: <one sentence explaining what was changed and why>"""


def suggest(
    clauses: list[Clause],
    risk_scores: list[RiskScore],
    groq_api_key: str = ""
) -> list[NegotiationSuggestion]:

    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("Warning: No Groq API key — skipping negotiation suggestions.")
        return []

    client = Groq(api_key=api_key)

    # Build a quick lookup of clause_id → clause
    clause_map = {c.clause_id: c for c in clauses}

    suggestions = []

    for risk in risk_scores:
        if risk.overall_risk not in SUGGEST_FOR:
            continue

        clause = clause_map.get(risk.clause_id)
        if not clause:
            continue

        prompt = NEGOTIATION_PROMPT.format(
            risk_level=risk.overall_risk.value.upper(),
            reasoning=risk.reasoning,
            clause_type=clause.clause_type.value if clause.clause_type else "unknown",
            clause_text=clause.text[:600]
        )

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.3
            )

            raw = response.choices[0].message.content.strip()
            suggested_text = ""
            rationale = ""

            for line in raw.splitlines():
                if line.startswith("SUGGESTED:"):
                    suggested_text = line.split(":", 1)[1].strip()
                elif line.startswith("RATIONALE:"):
                    rationale = line.split(":", 1)[1].strip()

            # Handle multi-line suggested text
            if not suggested_text:
                lines = raw.splitlines()
                for i, line in enumerate(lines):
                    if "SUGGESTED:" in line:
                        suggested_text = " ".join(lines[i:]).replace("SUGGESTED:", "").replace("RATIONALE:", "||").split("||")[0].strip()
                        break

            if suggested_text:
                suggestions.append(NegotiationSuggestion(
                    clause_id=clause.clause_id,
                    original_text=clause.text,
                    suggested_text=suggested_text,
                    rationale=rationale or "Rebalanced clause language."
                ))

        except Exception as e:
            print(f"  Warning: Could not generate suggestion for {clause.clause_id} — {e}")
            continue


    return suggestions
