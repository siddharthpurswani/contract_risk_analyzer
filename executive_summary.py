def build_executive_summary_prompt(clauses):
    key_points = []

    for c in clauses:
        if c.get("risk_level") in ["Medium", "High"]:
            category = c.get("category", "Unclassified")
            key_points.append(
                f"- {category} clause may require attention."
            )


    joined_points = "\n".join(key_points)

    return f"""
You are a business-focused legal assistant.

Based on the following findings, generate a concise executive summary
(4–5 bullet points, non-technical language).

Findings:
{joined_points}

Do not use legal jargon.
Do not add information not present in the findings.
"""


def build_risk_summary_prompt(clauses):
    risk_details = []

    # Clause-level risks
    for c in clauses:
        if c.get("risk_level") in ["Medium", "High"]:
            risk_details.append(
                f"""Clause Type: {c.get("category", "Unclassified")}
    Risk Level: {c["risk_level"]}
    Explanation: {c.get("llm_risk_explanation", c.get("risk_reason", "No explanation provided."))}
    """
            )


    joined_details = "\n".join(risk_details)

    return f"""You are a legal risk analyst.

Generate a precise contract-level risk summary highlighting
material risks based only on the information below.

Limit the answer to 200 words.

Risk Details:
{joined_details}

Rules:
- Do not introduce new risks
- Do not speculate beyond the provided information
- Use clear, lawyer-friendly language
"""


from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = OllamaLLM(model="llama3.2:1b")

def generate_summary(prompt: str) -> str:
    chain = llm | StrOutputParser()
    return chain.invoke(prompt)


def generate_contract_summaries(clauses):
    exec_prompt = build_executive_summary_prompt(clauses)
    risk_prompt = build_risk_summary_prompt(clauses)

    executive_summary = generate_summary(exec_prompt)
    risk_summary = generate_summary(risk_prompt)

    return {
        "executive_summary": executive_summary,
        "risk_summary": risk_summary
    }



