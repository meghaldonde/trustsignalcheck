import asyncio
import hashlib
import os
import secrets
import time
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request, status
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

# Salt for IP hashing (prevents rainbow table attacks)
IP_HASH_SALT = os.environ.get("IP_HASH_SALT", "signalcheck-default-salt-2024")


def hash_ip(ip_address: str) -> str:
    """
    Hash IP address using SHA-256 with salt.
    One-way hash - cannot be reversed to get original IP.
    """
    salted = f"{IP_HASH_SALT}:{ip_address}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]  # First 16 chars

# Gemini pricing (per 1M tokens) - Gemini 3.6 Flash
GEMINI_INPUT_COST_PER_1M = 0.075  # $0.075 per 1M input tokens
GEMINI_OUTPUT_COST_PER_1M = 0.30  # $0.30 per 1M output tokens

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


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on token usage."""
    input_cost = (input_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M
    return input_cost + output_cost


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
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
    input_tokens = len(prompt) // 4
    output_tokens = len(response_text) // 4

    return AIAnalysis.model_validate_json(response_text), input_tokens, output_tokens


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
    # Get client IP and hash it (privacy-preserving)
    client_ip = request.client.host if request.client else "unknown"
    # Check for forwarded IP (when behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

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
    domain_signal, (ai_analysis, input_tokens, output_tokens) = await asyncio.gather(
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
        url=scan_request.url,
        domain_signal_score=domain_signal.reputation_score,
        ai_probability_score=ai_analysis.ai_probability_score,
        signal_trust_score=signal_trust_score,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        response_time_ms=response_time_ms,
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
        h1 { margin-bottom: 24px; color: #1a1a1a; }
        h2 { margin: 24px 0 16px; color: #333; font-size: 18px; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
        .card-value.green { color: #34a853; }
        .card-value.blue { color: #4285f4; }
        .card-value.orange { color: #fbbc04; }
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
    </style>
</head>
<body>
    <h1>SignalCheck Admin Dashboard</h1>
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

    <h2>Overview</h2>
    <div class="grid" id="stats-grid">
        <div class="card"><div class="card-label">Loading...</div></div>
    </div>

    <h2>Daily Stats (Last 7 Days)</h2>
    <table id="daily-table">
        <thead><tr><th>Date</th><th>Scans</th><th>Tokens</th><th>Cost</th></tr></thead>
        <tbody><tr><td colspan="4">Loading...</td></tr></tbody>
    </table>

    <h2>Recent Scans</h2>
    <table id="recent-table">
        <thead><tr><th>Time</th><th>User</th><th>URL</th><th>Trust Score</th><th>AI Prob</th><th>Cost</th><th>Response</th><th>Action</th></tr></thead>
        <tbody><tr><td colspan="8">Loading...</td></tr></tbody>
    </table>

    <script>
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
                    <div class="card-label">Scans Today</div>
                    <div class="card-value">${data.scans_today}</div>
                </div>
                <div class="card">
                    <div class="card-label">Total Tokens</div>
                    <div class="card-value">${data.total_tokens.toLocaleString()}</div>
                </div>
                <div class="card">
                    <div class="card-label">Total Cost</div>
                    <div class="card-value green">$${data.total_cost_usd.toFixed(4)}</div>
                </div>
                <div class="card">
                    <div class="card-label">Cost Today</div>
                    <div class="card-value">$${data.cost_today_usd.toFixed(4)}</div>
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

            // Daily stats table
            const dailyBody = data.daily_stats.map(d => `
                <tr>
                    <td>${d.date}</td>
                    <td>${d.scans}</td>
                    <td>${(d.tokens || 0).toLocaleString()}</td>
                    <td>$${(d.cost || 0).toFixed(4)}</td>
                </tr>
            `).join('') || '<tr><td colspan="4">No data</td></tr>';
            document.querySelector('#daily-table tbody').innerHTML = dailyBody;

            // Recent scans table
            const recentBody = data.recent_scans.map(s => `
                <tr>
                    <td>${new Date(s.created_at).toLocaleString()}</td>
                    <td>${s.user_id.slice(0, 8)}...</td>
                    <td>${s.url.slice(0, 30)}...</td>
                    <td>${s.signal_trust_score}</td>
                    <td>${s.ai_probability_score}%</td>
                    <td>$${s.cost_usd.toFixed(4)}</td>
                    <td>${s.response_time_ms}ms</td>
                    <td><button class="grant-link" onclick="fillGrantForm('${s.user_id}')">Grant</button></td>
                </tr>
            `).join('') || '<tr><td colspan="8">No data</td></tr>';
            document.querySelector('#recent-table tbody').innerHTML = recentBody;

            // Allowances table
            const allowancesBody = (data.allowances || []).map(a => `
                <tr>
                    <td>${a.user_id.slice(0, 8)}...</td>
                    <td>${a.extra_scans}</td>
                    <td>${a.notes || '-'}</td>
                    <td>${new Date(a.granted_at).toLocaleString()}</td>
                </tr>
            `).join('') || '<tr><td colspan="4">No allowances granted</td></tr>';
            document.querySelector('#allowances-table tbody').innerHTML = allowancesBody;
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
