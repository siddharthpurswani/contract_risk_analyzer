MANDATORY_CLAUSES = [
    "Liability",
    "Termination",
    "Confidentiality",
    "Payment",
    "Indemnity",
    "Governing Law"
]

CONFIDENCE_THRESHOLD = 0.5  # avoid weak classifications

def get_present_clause_types(clauses):
    """
    Extract confidently detected clause categories.
    """
    return {
        c["category"]
        for c in clauses
        if c.get("category") not in (None, "Other")
        and c.get("classification_confidence", 1.0) >= CONFIDENCE_THRESHOLD
    }

def detect_missing_clauses(clauses):
    """
    Detect mandatory clauses missing from the contract.
    
    Input:
        clauses: list of clause dicts
    
    Output:
        list of missing clause findings
    """
    present = get_present_clause_types(clauses)
    missing = []

    for required in MANDATORY_CLAUSES:
        if required not in present:
            missing.append({
                "clause_type": required,
                "severity": "High",
                "risk_type": "Missing Clause",
                "reason": f"Mandatory {required} clause not found in contract"
            })

    return missing