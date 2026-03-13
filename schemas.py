from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class ClauseType(str, Enum):
    LIABILITY = "liability"
    INDEMNITY = "indemnity"
    TERMINATION = "termination"
    IP_OWNERSHIP = "ip_ownership"
    PAYMENT = "payment"
    CONFIDENTIALITY = "confidentiality"
    DISPUTE_RESOLUTION = "dispute_resolution"
    GOVERNING_LAW = "governing_law"
    FORCE_MAJEURE = "force_majeure"
    WARRANTY = "warranty"
    AMENDMENT = "amendment"
    ENTIRE_AGREEMENT = "entire_agreement"
    NON_COMPETE = "non_compete"
    PROFIT_SHARING = "profit_sharing"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Clause(BaseModel):
    clause_id: str
    heading: Optional[str]
    text: str
    start_line: int
    end_line: int
    confidence: float  # segmentation confidence 0.0 - 1.0
    clause_type: Optional[ClauseType] = ClauseType.UNKNOWN


class RiskScore(BaseModel):
    clause_id: str
    legal_risk: RiskLevel
    financial_risk: RiskLevel
    operational_risk: RiskLevel
    overall_risk: RiskLevel
    reasoning: str
    deviation_score: Optional[float] = None  # vs standard template (RAG)


class Conflict(BaseModel):
    clause_id_a: str
    clause_id_b: str
    conflict_description: str
    severity: RiskLevel


class NegotiationSuggestion(BaseModel):
    clause_id: str
    original_text: str
    suggested_text: str
    rationale: str


class AnalysisResult(BaseModel):
    document_type: str
    total_clauses: int
    clauses: List[Clause]
    risk_scores: List[RiskScore]
    conflicts: List[Conflict]
    suggestions: List[NegotiationSuggestion]
    business_summary: str
    legal_summary: str