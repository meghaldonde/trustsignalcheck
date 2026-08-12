import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import ValidationError
from pysafebrowsing import SafeBrowsing
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from google import genai

from database import (
    check_rate_limit,
    cleanup_old_data,
    get_admin_stats,
    get_all_allowances,
    get_or_create_user,
    grant_extra_scans,
    increment_scan_count,
    log_scan,
)
from schemas import AIAnalysis, DomainSignalScore, GrantScansRequest, ScanRequest, ScanResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Salt for IP hashing - REQUIRED, no default (security fix)
IP_HASH_SALT = os.environ.get("IP_HASH_SALT")
if not IP_HASH_SALT:
    raise RuntimeError("IP_HASH_SALT environment variable must be set")


def get_client_ip(request: Request) -> str:
    """
    Extract and validate client IP from request.
    On Render, the real client IP is appended to X-Forwarded-For (take last entry).
    Validates IP format to prevent header spoofing bypass.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    # Render appends the real client IP; take the LAST entry, not the first
    candidate = xff.split(",")[-1].strip() if xff else ""

    # Validate it's a real IP address (prevents spoofing with garbage)
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except ValueError:
        # Invalid or empty - fall back to direct connection
        return request.client.host if request.client else "unknown"


def hash_ip(ip_address: str) -> str:
    """
    Hash IP address using SHA-256 with salt.
    One-way hash - cannot be reversed to get original IP.
    """
    salted = f"{IP_HASH_SALT}:{ip_address}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]  # First 16 chars

# Gemini pricing (per 1M tokens) - Gemini 3.6 Flash
# Source: https://ai.google.dev/gemini-api/docs/pricing
# Verified 2026-08-12. Re-check before citing these figures anywhere public.
GEMINI_INPUT_COST_PER_1M = 1.50   # $1.50 per 1M input tokens
GEMINI_OUTPUT_COST_PER_1M = 7.50  # $7.50 per 1M output tokens

# Rate limit config
MAX_SCANS_PER_DAY = 1

# Admin credentials (set via environment variables)
# For local dev: export ADMIN_USERNAME=admin ADMIN_PASSWORD=yourpassword
# For Render: set in Environment tab
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

app = FastAPI(title="SignalCheck API")
security = HTTPBasic()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials using constant-time comparison."""
    # If credentials not configured, deny all access
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin credentials not configured",
        )

    correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        ADMIN_USERNAME.encode("utf-8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        ADMIN_PASSWORD.encode("utf-8")
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# CORS: Only allow requests from the Chrome extension
# Note: Extension fetches with host_permissions bypass page CORS anyway,
# but this is defense in depth for any browser-based testing
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^chrome-extension://[a-z]{32}$",  # Chrome extension IDs are 32 lowercase letters
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# Gemini client (lazy initialization)
_gemini_client = None

# Google Safe Browsing client (lazy initialization)
_safebrowsing_client = None
_executor = ThreadPoolExecutor(max_workers=2)

# Safe Browsing API key (separate from Gemini)
SAFE_BROWSING_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    return _gemini_client


def get_safebrowsing_client():
    global _safebrowsing_client
    if _safebrowsing_client is None and SAFE_BROWSING_API_KEY:
        _safebrowsing_client = SafeBrowsing(SAFE_BROWSING_API_KEY)
    return _safebrowsing_client


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on token usage."""
    input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M
    return input_cost + output_cost


def _check_url_with_safebrowsing(url: str) -> dict | None | str:
    """
    Synchronous Safe Browsing lookup (runs in thread pool).
    Returns dict on success, None if client unavailable, "error" string on API failure.
    """
    client = get_safebrowsing_client()
    if not client:
        return None
    try:
        result = client.lookup_urls([url])
        return result.get(url)
    except Exception as e:
        logger.error(f"Safe Browsing API error: {e}")
        return "error"  # Distinct from None (not configured) for monitoring


async def check_domain_signal(url: str) -> DomainSignalScore:
    """
    Check domain reputation using Google Safe Browsing API.
    Falls back to mock implementation if API key not configured.
    """
    # Try Google Safe Browsing API first
    if SAFE_BROWSING_API_KEY:
        loop = asyncio.get_running_loop()  # Fixed: use get_running_loop() not deprecated get_event_loop()
        result = await loop.run_in_executor(_executor, _check_url_with_safebrowsing, str(url))

        # Handle API error distinctly from "not configured"
        if result == "error":
            return DomainSignalScore(
                reputation_score=50,
                source="safe_browsing_unavailable",  # Distinct source for monitoring
                threat_type=None
            )

        if result is not None:
            if result.get("malicious"):
                # URL is flagged as dangerous
                threat_type = result.get("threats", [{}])[0].get("threatType", "UNKNOWN")
                return DomainSignalScore(
                    reputation_score=10,  # Very low trust
                    source="google_safe_browsing",
                    threat_type=threat_type
                )
            else:
                # URL passed Safe Browsing check - safe
                return DomainSignalScore(
                    reputation_score=85,
                    source="google_safe_browsing",
                    threat_type=None
                )

    # Fallback: Mock implementation if API key not set
    domain = urlparse(str(url)).netloc
    trusted_domains = ["google.com", "github.com", "wikipedia.org", "example.com"]

    if any(trusted in domain for trusted in trusted_domains):
        return DomainSignalScore(reputation_score=90, source="mock_trusted_list")

    return DomainSignalScore(reputation_score=50, source="mock_unknown")


async def analyze_with_gemini(text: str) -> tuple[AIAnalysis, int, int]:
    """
    Analyze text using Gemini Interactions API.
    Returns (AIAnalysis, input_tokens, output_tokens).
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

    # Strip markdown fences more robustly
    if response_text.startswith("```"):
        # Find the actual JSON content between fences
        lines = response_text.split("\n")
        json_lines = []
        in_json = False
        for line in lines:
            if line.startswith("```") and not in_json:
                in_json = True
                continue
            elif line.startswith("```") and in_json:
                break
            elif in_json:
                json_lines.append(line)
        response_text = "\n".join(json_lines)

    # Get real token counts from API response (not estimates)
    # interactions.create() uses .usage with total_input_tokens, total_output_tokens, total_thought_tokens
    usage = getattr(interaction, "usage", None)
    if usage:
        input_tokens = getattr(usage, "total_input_tokens", 0) or 0
        # Output cost includes thought/reasoning tokens (billed but not in output_text)
        output_tokens = (getattr(usage, "total_output_tokens", 0) or 0) + (getattr(usage, "total_thought_tokens", 0) or 0)
        token_source = "measured"
        logger.info(f"Token usage (measured): input={input_tokens}, output={output_tokens} (incl. {getattr(usage, 'total_thought_tokens', 0)} thought tokens)")
    else:
        # Fallback to estimate if usage not available
        input_tokens = len(prompt) // 4
        output_tokens = len(response_text) // 4
        token_source = "estimated"
        logger.warning(f"Token usage (estimated): input={input_tokens}, output={output_tokens}")

    # Parse and validate response with error handling
    try:
        analysis = AIAnalysis.model_validate_json(response_text)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse Gemini response: {e}. Raw response: {response_text[:500]}")
        raise HTTPException(
            status_code=502,
            detail="AI analysis returned invalid response. Please try again."
        )

    return analysis, input_tokens, output_tokens, token_source


def calculate_signal_trust_score(domain_signal: DomainSignalScore, ai_analysis: AIAnalysis) -> int:
    """
    Calculate Signal Trust Score.
    Higher domain signal + lower AI probability = higher trust.
    """
    domain_weight = 0.4
    content_weight = 0.6
    content_authenticity = 100 - ai_analysis.ai_probability_score
    combined = (domain_signal.reputation_score * domain_weight) + (content_authenticity * content_weight)
    return int(combined)


@app.post("/api/scan", response_model=ScanResponse)
async def scan(
    scan_request: ScanRequest,
    request: Request,
):
    """
    Scan a URL and text snippet for trust signals.
    Rate limited to 1 scan per user per day.
    User identified by hashed IP (privacy-preserving).
    """
    # Get client IP securely (validates to prevent spoofing)
    client_ip = get_client_ip(request)
    user_id = hash_ip(client_ip)
    get_or_create_user(user_id)

    # Cleanup old data (24hr retention policy)
    cleanup_old_data()

    # Check rate limit
    is_allowed, remaining = check_rate_limit(user_id, MAX_SCANS_PER_DAY)
    if not is_allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"You have reached the limit of {MAX_SCANS_PER_DAY} scan(s) per day.",
                "remaining_scans": 0,
            }
        )

    # Track response time
    start_time = time.time()

    # Run domain check and AI analysis in parallel
    domain_signal, (ai_analysis, input_tokens, output_tokens, token_source) = await asyncio.gather(
        check_domain_signal(scan_request.url),
        analyze_with_gemini(scan_request.text_snippet),
    )

    signal_trust_score = calculate_signal_trust_score(domain_signal, ai_analysis)

    # Calculate metrics
    response_time_ms = int((time.time() - start_time) * 1000)
    cost_usd = calculate_cost(input_tokens, output_tokens)

    # Log scan and increment rate limit counter
    log_scan(
        user_id=user_id,
        url=str(scan_request.url),  # Convert HttpUrl to str for sqlite
        domain_signal_score=domain_signal.reputation_score,
        ai_probability_score=ai_analysis.ai_probability_score,
        signal_trust_score=signal_trust_score,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        response_time_ms=response_time_ms,
        token_source=token_source,
    )
    increment_scan_count(user_id)

    return ScanResponse(
        domain_signal_score=domain_signal,
        ai_analysis=ai_analysis,
        signal_trust_score=signal_trust_score,
    )


@app.get("/api/admin/stats")
async def admin_stats(username: str = Depends(verify_admin)):
    """Get admin dashboard statistics. Requires authentication."""
    stats = get_admin_stats()
    stats["allowances"] = get_all_allowances()
    return stats


@app.post("/api/admin/grant-scans")
async def admin_grant_scans(
    request: GrantScansRequest,
    username: str = Depends(verify_admin)
):
    """Grant extra daily scans to a user. Requires authentication."""
    if request.extra_scans < 0:
        raise HTTPException(
            status_code=400,
            detail="extra_scans must be non-negative"
        )
    return grant_extra_scans(
        user_id=request.user_id,
        extra_scans=request.extra_scans,
        notes=request.notes
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(username: str = Depends(verify_admin)):
    """Render admin dashboard HTML. Requires authentication."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignalCheck Admin Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 24px;
            color: #333;
        }
        h1 { margin-bottom: 24px; color: #1a1a1a; display: flex; align-items: center; gap: 12px; }
        h1 svg { width: 32px; height: 32px; }
        h2 { margin: 24px 0 16px; color: #333; font-size: 18px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .card.accent {
            border-left: 4px solid #fbbc04;
        }
        .card-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .card-value {
            font-size: 32px;
            font-weight: bold;
            margin-top: 8px;
            color: #1a1a1a;
        }
        .card-sub {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        .card-value.green { color: #34a853; }
        .card-value.blue { color: #4285f4; }
        .card-value.orange { color: #fbbc04; }
        .cost-note {
            font-size: 12px;
            color: #666;
            font-style: italic;
            margin-bottom: 24px;
        }
        table {
            width: 100%;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #fafafa;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
        }
        td { font-size: 14px; }
        .refresh-btn {
            background: #4285f4;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin-bottom: 24px;
        }
        .refresh-btn:hover { background: #3367d6; }
        .loading { opacity: 0.5; }
        .grant-form {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 24px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: flex-end;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .form-group label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .form-group input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
        }
        .grant-btn {
            background: #34a853;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .grant-btn:hover { background: #2d9249; }
        .message {
            padding: 10px 16px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: none;
        }
        .message.success { background: #e6f4ea; color: #1e7e34; display: block; }
        .message.error { background: #fce8e6; color: #c5221f; display: block; }
        .grant-link {
            background: #4285f4;
            color: white;
            border: none;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .grant-link:hover { background: #3367d6; }
        .badge {
            display: inline-block;
            font-size: 10px;
            padding: 1px 4px;
            border-radius: 3px;
            vertical-align: middle;
            margin-left: 2px;
        }
        .badge-measured {
            background: #d4edda;
            color: #155724;
        }
        .badge-estimated {
            background: #fff3cd;
            color: #856404;
        }
        .chart-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 24px;
        }
        .chart-empty {
            text-align: center;
            color: #666;
            padding: 40px;
        }
        .tooltip {
            cursor: help;
            border-bottom: 1px dotted #666;
        }
    </style>
</head>
<body>
    <h1><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img" aria-label="SignalCheck"><defs><linearGradient id="sc-shield" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4F46E5"/><stop offset="1" stop-color="#312E81"/></linearGradient><clipPath id="sc-clip"><path d="M 40 0 H 472 A 40 40 0 0 1 512 40 V 272 C 512 388 420 470 256 512 C 92 470 0 388 0 272 V 40 A 40 40 0 0 1 40 0 Z"/></clipPath></defs><path d="M 40 0 H 472 A 40 40 0 0 1 512 40 V 272 C 512 388 420 470 256 512 C 92 470 0 388 0 272 V 40 A 40 40 0 0 1 40 0 Z" fill="url(#sc-shield)"/><g clip-path="url(#sc-clip)" fill="none" stroke-width="64" stroke-linecap="round" stroke-linejoin="round"><path d="M 128 252 L 196 252 L 250 348 L 322 152 L 356 202" stroke="#FFFFFF"/><path d="M 356 202 L 384 252" stroke="#22D3EE"/></g></svg>SignalCheck Admin Dashboard</h1>
    <button class="refresh-btn" onclick="loadStats()">Refresh Data</button>

    <h2>Grant Extra Scans</h2>
    <div id="grant-message" class="message"></div>
    <div class="grant-form">
        <div class="form-group">
            <label>User ID (hash)</label>
            <input type="text" id="grant-user-id" placeholder="e.g. a1b2c3d4e5f6" style="width: 180px;">
        </div>
        <div class="form-group">
            <label>Extra Scans</label>
            <input type="number" id="grant-extra-scans" value="5" min="0" style="width: 80px;">
        </div>
        <div class="form-group">
            <label>Notes (optional)</label>
            <input type="text" id="grant-notes" placeholder="Reason" style="width: 150px;">
        </div>
        <button class="grant-btn" onclick="grantScans()">Grant Scans</button>
    </div>

    <h2>Current Allowances</h2>
    <table id="allowances-table">
        <thead><tr><th>User ID</th><th>Extra Scans</th><th>Notes</th><th>Granted At</th></tr></thead>
        <tbody><tr><td colspan="4">Loading...</td></tr></tbody>
    </table>

    <h2>Cost Overview</h2>
    <div class="grid" id="stats-grid">
        <div class="card"><div class="card-label">Loading...</div></div>
    </div>
    <div class="cost-note" id="cost-note"></div>

    <h2>Daily Cost (Last 7 Days)</h2>
    <div class="chart-container" id="daily-chart">
        <div class="chart-empty">Loading...</div>
    </div>

    <h2>Recent Scans</h2>
    <table id="recent-table">
        <thead><tr><th>Time</th><th>User</th><th>URL</th><th>Trust Score</th><th>AI Prob</th><th>Cost</th><th>Response</th><th>Action</th></tr></thead>
        <tbody><tr><td colspan="8">Loading...</td></tr></tbody>
    </table>

    <script>
        // HTML escape function to prevent XSS
        const esc = s => String(s).replace(/[&<>"']/g,
            c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

        // Pricing constants - keep in sync with main.py
        const PRICE_IN  = 1.50;   // per 1M tokens
        const PRICE_OUT = 7.50;   // per 1M tokens
        const DAILY_BUDGET_USD = 1.00;

        async function loadStats() {
            document.body.classList.add('loading');
            try {
                const res = await fetch('/api/admin/stats');
                const data = await res.json();
                renderStats(data);
            } catch (e) {
                console.error(e);
            }
            document.body.classList.remove('loading');
        }

        function renderStats(data) {
            // Compute costs from token counts at current prices
            // (Stored cost_usd fields used old prices, so we ignore them)
            const inputCost  = (data.total_input_tokens / 1e6) * PRICE_IN;
            const outputCost = (data.total_output_tokens / 1e6) * PRICE_OUT;
            const totalCost  = inputCost + outputCost;
            const totalTokens = data.total_input_tokens + data.total_output_tokens;

            const inputPct  = totalCost > 0 ? Math.round((inputCost / totalCost) * 100) : 0;
            const outputPct = totalCost > 0 ? 100 - inputPct : 0;

            // Use valid_scans_for_cost (excludes zero-token rows) for accurate cost per scan
            const validScans = data.valid_scans_for_cost || data.total_scans;
            const costPerScan = validScans > 0 ? totalCost / validScans : 0;
            const headroom = costPerScan > 0 ? Math.floor(DAILY_BUDGET_USD / costPerScan) : null;

            // Provenance stats
            const measuredScans = data.measured_scans || 0;
            const estimatedScans = data.estimated_scans || 0;
            const zeroTokenScans = data.zero_token_scans || 0;

            // Blended rate for daily stats (tokens → cost)
            // Assumes each day had roughly the same input/output ratio as overall
            const blendedRate = totalTokens > 0 ? totalCost / totalTokens : 0;
            const dailyCost = d => ((d.tokens || 0) * blendedRate);

            // Run rate: average daily cost across days present, multiplied by 30
            let runRate = null;
            if (data.daily_stats && data.daily_stats.length > 0) {
                const totalDailyCost = data.daily_stats.reduce((sum, d) => sum + dailyCost(d), 0);
                const avgDailyCost = totalDailyCost / data.daily_stats.length;
                runRate = avgDailyCost * 30;
            }

            document.getElementById('stats-grid').innerHTML = `
                <div class="card">
                    <div class="card-label">Total Users</div>
                    <div class="card-value blue">${data.total_users}</div>
                </div>
                <div class="card">
                    <div class="card-label">Total Scans</div>
                    <div class="card-value blue">${data.total_scans}</div>
                </div>
                <div class="card">
                    <div class="card-label">Input Tokens</div>
                    <div class="card-value">${data.total_input_tokens.toLocaleString()}</div>
                    <div class="card-sub">$${inputCost.toFixed(4)} (${inputPct}% of cost)</div>
                </div>
                <div class="card accent">
                    <div class="card-label">Output Tokens</div>
                    <div class="card-value orange">${data.total_output_tokens.toLocaleString()}</div>
                    <div class="card-sub">$${outputCost.toFixed(4)} (${outputPct}% of cost)</div>
                </div>
                <div class="card">
                    <div class="card-label">Total Cost</div>
                    <div class="card-value green">$${totalCost.toFixed(4)}</div>
                </div>
                <div class="card">
                    <div class="card-label">Cost per Scan</div>
                    <div class="card-value">${costPerScan > 0 ? '$' + costPerScan.toFixed(4) : '--'}</div>
                </div>
                <div class="card">
                    <div class="card-label"><span class="tooltip" title="Assumes each day has the same input/output ratio as the overall period">Run Rate (est.)</span></div>
                    <div class="card-value">${runRate !== null ? '$' + runRate.toFixed(2) + '/mo' : '--'}</div>
                </div>
                <div class="card">
                    <div class="card-label">Users/day within $${DAILY_BUDGET_USD.toFixed(2)}/day</div>
                    <div class="card-value green">${headroom !== null ? headroom.toLocaleString() : '--'}</div>
                </div>
                <div class="card">
                    <div class="card-label">Avg Response Time</div>
                    <div class="card-value">${data.avg_response_time_ms}ms</div>
                </div>
                <div class="card">
                    <div class="card-label">Avg Trust Score</div>
                    <div class="card-value green">${data.avg_trust_score}</div>
                </div>
                <div class="card">
                    <div class="card-label">Avg AI Probability</div>
                    <div class="card-value orange">${data.avg_ai_probability}%</div>
                </div>
            `;

            // Provenance note.
            // measured + estimated partitions every scan. zeroTokenScans is a
            // SUBSET of those (a zero-token row still carries a token_source),
            // so it is phrased as "of which" rather than listed alongside.
            let provenanceNote = `Costs are computed from token counts at current published prices.`;
            if (measuredScans > 0 || estimatedScans > 0) {
                provenanceNote += ` Token data: ${measuredScans} measured, ${estimatedScans} estimated`;
                if (zeroTokenScans > 0) {
                    provenanceNote += ` (of which ${zeroTokenScans} had no token data and are excluded from cost)`;
                }
                provenanceNote += '.';
            }
            document.getElementById('cost-note').innerHTML = provenanceNote;

            // Daily cost chart (inline SVG)
            renderDailyChart(data.daily_stats, dailyCost);

            // Recent scans table (escape user-supplied data to prevent XSS)
            // Compute per-row cost from tokens at current prices, show provenance badge
            const recentBody = data.recent_scans.map(s => {
                const scanCost = ((s.input_tokens || 0) / 1e6) * PRICE_IN
                              + ((s.output_tokens || 0) / 1e6) * PRICE_OUT;
                const isMeasured = s.token_source === 'measured';
                const badge = isMeasured
                    ? '<span class="badge badge-measured" title="Token counts from API">✓</span>'
                    : '<span class="badge badge-estimated" title="Token counts estimated">~</span>';
                const costDisplay = scanCost > 0 ? '$' + scanCost.toFixed(4) + ' ' + badge : '--';
                return `
                <tr>
                    <td>${esc(new Date(s.created_at).toLocaleString())}</td>
                    <td>${esc(s.user_id.slice(0, 8))}...</td>
                    <td>${esc(s.url.slice(0, 30))}...</td>
                    <td>${esc(s.signal_trust_score)}</td>
                    <td>${esc(s.ai_probability_score)}%</td>
                    <td>${costDisplay}</td>
                    <td>${esc(s.response_time_ms)}ms</td>
                    <td><button class="grant-link" onclick="fillGrantForm('${esc(s.user_id)}')">Grant</button></td>
                </tr>
            `}).join('') || '<tr><td colspan="8">No data</td></tr>';
            document.querySelector('#recent-table tbody').innerHTML = recentBody;

            // Allowances table (escape user-supplied data to prevent XSS)
            const allowancesBody = (data.allowances || []).map(a => `
                <tr>
                    <td>${esc(a.user_id.slice(0, 8))}...</td>
                    <td>${esc(a.extra_scans)}</td>
                    <td>${esc(a.notes || '-')}</td>
                    <td>${esc(new Date(a.granted_at).toLocaleString())}</td>
                </tr>
            `).join('') || '<tr><td colspan="4">No allowances granted</td></tr>';
            document.querySelector('#allowances-table tbody').innerHTML = allowancesBody;
        }

        function renderDailyChart(dailyStats, dailyCostFn) {
            const container = document.getElementById('daily-chart');

            if (!dailyStats || dailyStats.length < 2) {
                container.innerHTML = '<div class="chart-empty">Not enough data yet (need at least 2 days)</div>';
                return;
            }

            // Reverse to show oldest to newest (left to right)
            const days = [...dailyStats].reverse();
            const costs = days.map(d => dailyCostFn(d));
            const maxCost = Math.max(...costs, 0.001); // Avoid division by zero

            const width = 600;
            const height = 200;
            const padding = { top: 20, right: 20, bottom: 40, left: 60 };
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;
            const barWidth = Math.min(60, (chartWidth / days.length) - 10);
            const barGap = (chartWidth - (barWidth * days.length)) / (days.length + 1);

            let svg = `<svg width="100%" viewBox="0 0 ${width} ${height}" style="max-width: ${width}px;">`;

            // Y-axis
            svg += `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${height - padding.bottom}" stroke="#ccc" stroke-width="1"/>`;

            // Y-axis labels
            const yTicks = 4;
            for (let i = 0; i <= yTicks; i++) {
                const y = padding.top + (chartHeight * i / yTicks);
                const val = maxCost * (1 - i / yTicks);
                svg += `<text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#666">$${val.toFixed(3)}</text>`;
                svg += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#eee" stroke-width="1"/>`;
            }

            // Bars
            days.forEach((d, i) => {
                const cost = costs[i];
                const barHeight = (cost / maxCost) * chartHeight;
                const x = padding.left + barGap + i * (barWidth + barGap);
                const y = padding.top + chartHeight - barHeight;

                svg += `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="#4285f4" rx="2"/>`;
                svg += `<text x="${x + barWidth/2}" y="${y - 5}" text-anchor="middle" font-size="10" fill="#333">${d.scans}</text>`;
                svg += `<text x="${x + barWidth/2}" y="${height - padding.bottom + 15}" text-anchor="middle" font-size="10" fill="#666">${d.date.slice(5)}</text>`;
            });

            svg += '</svg>';
            container.innerHTML = svg;
        }

        function fillGrantForm(userId) {
            document.getElementById('grant-user-id').value = userId;
            document.getElementById('grant-user-id').scrollIntoView({ behavior: 'smooth' });
        }

        async function grantScans() {
            const userId = document.getElementById('grant-user-id').value.trim();
            const extraScans = parseInt(document.getElementById('grant-extra-scans').value);
            const notes = document.getElementById('grant-notes').value.trim();
            const msgEl = document.getElementById('grant-message');

            if (!userId) {
                msgEl.className = 'message error';
                msgEl.textContent = 'Please enter a User ID';
                return;
            }

            try {
                const res = await fetch('/api/admin/grant-scans', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, extra_scans: extraScans, notes: notes })
                });

                if (res.ok) {
                    msgEl.className = 'message success';
                    msgEl.textContent = `Granted ${extraScans} extra scans to user ${userId.slice(0, 8)}...`;
                    document.getElementById('grant-user-id').value = '';
                    document.getElementById('grant-notes').value = '';
                    loadStats();
                } else {
                    const err = await res.json();
                    msgEl.className = 'message error';
                    msgEl.textContent = err.detail || 'Failed to grant scans';
                }
            } catch (e) {
                msgEl.className = 'message error';
                msgEl.textContent = 'Error: ' + e.message;
            }
        }

        loadStats();
        setInterval(loadStats, 30000); // Auto-refresh every 30s
    </script>
</body>
</html>
"""


@app.get("/health")
async def health():
    return {"status": "ok"}
