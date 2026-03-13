import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from segmenter import Clause
from schemas import ClauseType

# --- Keyword map (fast path) ---

CLAUSE_KEYWORDS = {
    ClauseType.LIABILITY: [
        "liability", "liable", "consequential damages", "indirect damages",
        "limitation of liability", "aggregate liability"
    ],
    ClauseType.INDEMNITY: [
        "indemnify", "indemnification", "hold harmless", "indemnity"
    ],
    ClauseType.TERMINATION: [
        "termination", "terminate", "notice of termination", "expiration",
        "termination for cause", "termination for convenience"
    ],
    ClauseType.CONFIDENTIALITY: [
        "confidential", "confidentiality", "non-disclosure", "proprietary information",
        "trade secret"
    ],
    ClauseType.IP_OWNERSHIP: [
        "intellectual property", "ip ownership", "copyright", "patent",
        "jointly developed", "pre-existing ip"
    ],
    ClauseType.PAYMENT: [
        "payment", "invoice", "fee", "pricing", "penalty", "late payment"
    ],
    ClauseType.DISPUTE_RESOLUTION: [
        "dispute", "arbitration", "mediation", "good faith negotiation",
        "dispute resolution"
    ],
    ClauseType.GOVERNING_LAW: [
        "governing law", "jurisdiction", "construed in accordance",
        "courts of", "applicable law"
    ],
    ClauseType.FORCE_MAJEURE: [
        "force majeure", "act of god", "beyond reasonable control",
        "natural disaster", "pandemic"
    ],
    ClauseType.WARRANTY: [
        "warranty", "warrant", "representation", "as is", "disclaimer"
    ],
    ClauseType.AMENDMENT: [
        "amendment", "amend", "modify", "modification", "written instrument"
    ],
    ClauseType.ENTIRE_AGREEMENT: [
        "entire agreement", "supersedes", "whole agreement", "prior negotiations"
    ],
    ClauseType.NON_COMPETE: [
        "non-compete", "non compete", "restraint of trade", "competing business"
    ],
    ClauseType.PROFIT_SHARING: [
        "profit sharing", "profit split", "revenue share", "net profits"
    ],
}


def classify_by_keywords(clause: Clause) -> tuple[ClauseType, float]:
    """Returns (ClauseType, confidence). Confidence 1.0 if heading matches, 0.8 if body matches."""
    text_lower = clause.text.lower()
    heading_lower = (clause.heading or "").lower()

    scores = {ctype: 0 for ctype in CLAUSE_KEYWORDS}

    for ctype, keywords in CLAUSE_KEYWORDS.items():
        for kw in keywords:
            if kw in heading_lower:
                scores[ctype] += 3   # heading match weighs more
            elif kw in text_lower:
                scores[ctype] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return ClauseType.UNKNOWN, 0.0
    confidence = 1.0 if scores[best] >= 3 else 0.8
    return best, confidence


def classify_by_llm(clause: Clause, client: Groq) -> ClauseType:
    """Fallback: ask Groq to classify the clause."""
    valid_types = [t.value for t in ClauseType if t != ClauseType.UNKNOWN]

    prompt = f"""You are a legal contract analyst. Classify the following contract clause into exactly one of these types:
{", ".join(valid_types)}

Clause heading: {clause.heading or "None"}
Clause text: {clause.text[:500]}

Respond with only the clause type, nothing else. Example: liability"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    result = response.choices[0].message.content.strip().lower()

    # Match to enum
    for ctype in ClauseType:
        if ctype.value == result:
            return ctype

    return ClauseType.UNKNOWN


# --- Main entry point ---

def classify(clauses: list[Clause], groq_api_key: str = "") -> list[Clause]:
    client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY", "")) if groq_api_key or os.getenv("GROQ_API_KEY") else None

    for clause in clauses:
        ctype, confidence = classify_by_keywords(clause)

        if ctype == ClauseType.UNKNOWN and client:
            # Fallback to LLM for ambiguous clauses
            ctype = classify_by_llm(clause, client)

        clause.clause_type = ctype


    return clauses
