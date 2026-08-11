from pydantic import BaseModel


class ScanRequest(BaseModel):
    url: str
    text_snippet: str


class AIAnalysis(BaseModel):
    ai_probability_score: int  # 0-100
    reasoning_flag: str
    key_signals: list[str]


class DomainSignalScore(BaseModel):
    reputation_score: int  # 0-100
    source: str
    threat_type: str | None = None  # e.g., "MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"


class ScanResponse(BaseModel):
    domain_signal_score: DomainSignalScore
    ai_analysis: AIAnalysis
    signal_trust_score: int  # 0-100


class GrantScansRequest(BaseModel):
    user_id: str
    extra_scans: int  # Total extra scans to grant (not additive)
    notes: str = ""
