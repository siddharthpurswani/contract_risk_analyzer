import re
import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -----------------------------
# 1️⃣ Static risk signals
# -----------------------------
RISK_KEYWORDS = {
    "Liability": ["unlimited liability", "no cap", "indemnify all"],
    "Termination": ["immediate termination", "no notice", "one-sided"],
    "Confidentiality": ["disclose freely", "no obligation", "vague"],
    "Payment": ["late payment unlimited", "no penalty", "deferred"],
    "Indemnity": ["indemnify completely", "unlimited obligation"],
    "Governing Law": ["jurisdiction unclear", "arbitration waived"],
    "Intellectual Property": ["IP ownership unclear", "work product assigned without consent"]
}

RISK_SCORE = {
    "Low": 0.2,
    "Medium": 0.5,
    "High": 0.8
}

# -----------------------------
# 2️⃣ LLM setup
# -----------------------------
llm = OllamaLLM(model="llama3.2:1b", temperature=0)

def llm_evaluate_risk(clause_text, category):
    """
    Use LLM to assess subtle business/legal risk.
    Returns a dict: {"risk_level": str, "risk_reason": str}
    """
    prompt = PromptTemplate(
        input_variables=[],
        template=f"""
        You are a legal risk analyst. Evaluate the following contract clause for potential business/legal risk.

        Clause Category: {category}
        Clause Text: {clause_text}

        1. Assign a risk level: Low, Medium, or High
        2. Explain why this clause might be risky (brief explanation)
        3. Return only a JSON object with keys: 'risk_level', 'risk_reason'
        """
    )

    response = llm.invoke(prompt.format())
    
    # Extract JSON from LLM response
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {"risk_level": "Low", "risk_reason": "LLM did not return valid JSON"}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"risk_level": "Low", "risk_reason": "LLM returned invalid JSON"}

# -----------------------------
# 3️⃣ Static + deviation rules
# -----------------------------
def detect_risk_static(clause):
    """
    Assign risk based on template deviation + keywords
    """
    text = clause["text"].lower()
    cat = clause.get("category", "Other")
    deviation = clause.get("is_deviation", False)

    risk_level = "Low"
    risk_reason = []

    

    # Keyword signals
    keywords = RISK_KEYWORDS.get(cat, [])
    for kw in keywords:
        if kw.lower() in text:
            risk_level = "High"
            risk_reason.append(f"Contains risky phrase: '{kw}'")

    return {
        "risk_level": risk_level,
        "risk_reason": "; ".join(risk_reason) if risk_reason else "No obvious risk"
    }

# -----------------------------
# 4️⃣ Hybrid risk evaluation
# -----------------------------
def detect_risk(clause):
    """
    Hybrid risk evaluation:
    - static rules
    - deviation
    - LLM evaluation
    """
    static_result = detect_risk_static(clause)
    llm_result = llm_evaluate_risk(clause["text"], clause.get("category", "Other"))

    # Pick higher risk level
    levels = ["Low", "Medium", "High"]
    static_idx = levels.index(static_result["risk_level"])
    llm_idx = levels.index(llm_result["risk_level"])
    final_idx = max(static_idx, llm_idx)

    clause["risk_level"] = levels[final_idx]
    clause["risk_score"] = RISK_SCORE[levels[final_idx]]
    clause["risk_reason"] = "; ".join(filter(None, [static_result["risk_reason"], llm_result["risk_reason"]]))

    return clause

def detect_risks(clauses):
    return [detect_risk(c) for c in clauses]

# -----------------------------
# 5️⃣ Example usage
# -----------------------------
if __name__ == "__main__":
    test_clauses = [
        {
            "clause_text": "The parties shall not be liable for any indirect damages, except for gross negligence.",
            "category": "Liability",
            "template_similarity": 0.68,
            "is_deviation": True
        },
        {
            "clause_text": "This agreement may be terminated by either party with thirty (30) days written notice.",
            "category": "Termination",
            "template_similarity": 0.82,
            "is_deviation": False
        }
    ]

    results = detect_risks(test_clauses)

    for c in results:
        print(f"Clause: {c['text']}")
        print(f"Risk Level: {c['risk_level']}")
        print(f"Score: {c['risk_score']}")
        print(f"Reason: {c['risk_reason']}")
        print("------")
