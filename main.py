from clause_segmentation import extract_clauses
from clause_classifier import ClauseClassifier
from rag_template import compare_with_templates
from risk_score import detect_risks
from executive_summary import generate_contract_summaries
from rag_template import TEMPLATE_DIR


contract_text = """
MASTER SERVICES AGREEMENT

1. PAYMENT TERMS
Payment shall be made within thirty (30) days from the invoice date. Late payments shall accrue interest at the rate of 1.5% per month. All amounts shall be paid in USD via wire transfer.

2. SCOPE OF SERVICES
The Service Provider shall deliver data analytics and reporting services as detailed in Schedule A. Services shall be performed in a professional and workmanlike manner.

3. CONFIDENTIALITY
Each party agrees to keep confidential all non-public, proprietary, or confidential information received from the other party. Confidentiality obligations shall survive termination of this Agreement for a period of three (3) years.

4. TERMINATION
Either party may terminate this Agreement for convenience by providing sixty (60) days’ prior written notice. Either party may terminate immediately in the event of a material breach that remains uncured for thirty (30) days.

5. LIMITATION OF LIABILITY
In no event shall either party be liable for indirect, incidental, or consequential damages. The total aggregate liability of either party shall not exceed the total fees paid under this Agreement.

6. INDEMNIFICATION
The Service Provider shall indemnify and hold harmless the Client against any third-party claims arising from the Service Provider’s negligence or willful misconduct.

7. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to conflict of laws principles.

8. FORCE MAJEURE
Neither party shall be liable for delays or failure to perform due to events beyond reasonable control, including acts of God, war, or governmental actions.
"""

def run_contract_analysis(contract_text: str):
    clauses = extract_clauses(contract_text)
    classifier = ClauseClassifier()
    clauses = classifier.classify_clauses(clauses)
    
    clauses = compare_with_templates(clauses, TEMPLATE_DIR)
    clauses = detect_risks(clauses)
    result = generate_contract_summaries(clauses)

    return result


