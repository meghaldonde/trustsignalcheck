import asyncio
import os
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from schemas import AIAnalysis, DomainSignalScore, ScanRequest, ScanResponse

app = FastAPI(title="SignalCheck API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini client (lazy initialization)
_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _gemini_client


async def check_domain_signal(url: str) -> DomainSignalScore:
    """
    Placeholder for domain reputation check.
    Can be replaced with Google Safe Browsing or VirusTotal API.
    """
    domain = urlparse(url).netloc

    # Mock implementation - returns high trust for known domains
    trusted_domains = ["google.com", "github.com", "wikipedia.org", "example.com"]

    if any(trusted in domain for trusted in trusted_domains):
        return DomainSignalScore(reputation_score=90, source="mock_trusted_list")

    # Default moderate trust for unknown domains
    return DomainSignalScore(reputation_score=50, source="mock_unknown")


async def analyze_with_gemini(text: str) -> AIAnalysis:
    """
    Analyze text using Gemini Interactions API.
    """
    prompt = f"""Analyze the following text and determine if it appears to be AI-generated or human-written.
Return ONLY valid JSON with these exact fields:
- ai_probability_score: integer 0-100 (0 = definitely human, 100 = definitely AI)
- reasoning_flag: brief explanation string
- key_signals: array of strings with specific signals noticed

Text to analyze:
{text[:2000]}

Respond with JSON only, no markdown or extra text."""

    client = get_gemini_client()
    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    # Parse JSON response
    response_text = interaction.output_text.strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    return AIAnalysis.model_validate_json(response_text)


def calculate_signal_trust_score(domain_signal: DomainSignalScore, ai_analysis: AIAnalysis) -> int:
    """
    Calculate Signal Trust Score.
    Higher domain signal + lower AI probability = higher trust.
    """
    # Weight: 40% domain signal, 60% content authenticity
    domain_weight = 0.4
    content_weight = 0.6

    # Invert AI probability (100 - score) since lower AI probability means more trustworthy
    content_authenticity = 100 - ai_analysis.ai_probability_score

    combined = (domain_signal.reputation_score * domain_weight) + (content_authenticity * content_weight)
    return int(combined)


@app.post("/api/scan", response_model=ScanResponse)
async def scan(request: ScanRequest):
    """
    Scan a URL and text snippet for trust signals.
    Runs domain check and AI analysis in parallel for better latency.
    """
    domain_signal, ai_analysis = await asyncio.gather(
        check_domain_signal(request.url),
        analyze_with_gemini(request.text_snippet),
    )

    signal_trust_score = calculate_signal_trust_score(domain_signal, ai_analysis)

    return ScanResponse(
        domain_signal_score=domain_signal,
        ai_analysis=ai_analysis,
        signal_trust_score=signal_trust_score,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
