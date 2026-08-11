import sqlite3
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent / "signalcheck.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            url TEXT NOT NULL,
            domain_signal_score INTEGER,
            ai_probability_score INTEGER,
            signal_trust_score INTEGER,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cost_usd REAL DEFAULT 0.0,
            response_time_ms INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS daily_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            scan_date DATE NOT NULL,
            scan_count INTEGER DEFAULT 0,
            UNIQUE(user_id, scan_date)
        );

        -- Cumulative aggregates table (no PII, stores historical stats)
        CREATE TABLE IF NOT EXISTS daily_aggregates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agg_date DATE UNIQUE NOT NULL,
            total_scans INTEGER DEFAULT 0,
            unique_users INTEGER DEFAULT 0,
            total_input_tokens INTEGER DEFAULT 0,
            total_output_tokens INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0.0,
            avg_response_time_ms REAL DEFAULT 0.0,
            avg_trust_score REAL DEFAULT 0.0,
            avg_ai_probability REAL DEFAULT 0.0,
            avg_domain_score REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Admin-granted scan allowances
        CREATE TABLE IF NOT EXISTS scan_allowances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            extra_scans INTEGER DEFAULT 0,
            notes TEXT,
            granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans(user_id);
        CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);
        CREATE INDEX IF NOT EXISTS idx_daily_limits_user_date ON daily_limits(user_id, scan_date);
        CREATE INDEX IF NOT EXISTS idx_daily_aggregates_date ON daily_aggregates(agg_date);
        CREATE INDEX IF NOT EXISTS idx_scan_allowances_user ON scan_allowances(user_id);
    """)

    conn.commit()
    conn.close()


def get_or_create_user(user_id: str) -> dict:
    """Get or create a user by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()

    if not user:
        cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cursor.fetchone()

    conn.close()
    return dict(user)


def get_user_scan_allowance(user_id: str) -> int:
    """Get extra scans granted to user by admin."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT extra_scans FROM scan_allowances WHERE user_id = ?",
        (user_id,)
    )
    result = cursor.fetchone()
    conn.close()
    return result["extra_scans"] if result else 0


def check_rate_limit(user_id: str, max_scans_per_day: int = 1) -> tuple[bool, int]:
    """
    Check if user has exceeded daily rate limit.
    Considers admin-granted extra scans.
    Returns (is_allowed, remaining_scans).
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()

    # Get base limit + any admin-granted extra scans
    extra_scans = get_user_scan_allowance(user_id)
    total_allowed = max_scans_per_day + extra_scans

    cursor.execute(
        "SELECT scan_count FROM daily_limits WHERE user_id = ? AND scan_date = ?",
        (user_id, today)
    )
    result = cursor.fetchone()

    if result:
        scan_count = result["scan_count"]
        remaining = max(0, total_allowed - scan_count)
        is_allowed = scan_count < total_allowed
    else:
        is_allowed = True
        remaining = total_allowed

    conn.close()
    return is_allowed, remaining


def increment_scan_count(user_id: str):
    """Increment daily scan count for user."""
    conn = get_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()

    cursor.execute("""
        INSERT INTO daily_limits (user_id, scan_date, scan_count)
        VALUES (?, ?, 1)
        ON CONFLICT(user_id, scan_date)
        DO UPDATE SET scan_count = scan_count + 1
    """, (user_id, today))

    conn.commit()
    conn.close()


def grant_extra_scans(user_id: str, extra_scans: int, notes: str = "") -> dict:
    """
    Grant extra daily scans to a user (admin function).
    Sets the total extra scans (not additive).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scan_allowances (user_id, extra_scans, notes)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET extra_scans = ?, notes = ?, granted_at = CURRENT_TIMESTAMP
    """, (user_id, extra_scans, notes, extra_scans, notes))

    conn.commit()
    conn.close()

    return {"user_id": user_id, "extra_scans": extra_scans, "notes": notes}


def get_all_allowances() -> list[dict]:
    """Get all users with scan allowances."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT user_id, extra_scans, notes, granted_at
        FROM scan_allowances
        WHERE extra_scans > 0
        ORDER BY granted_at DESC
    """)

    allowances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return allowances


def log_scan(
    user_id: str,
    url: str,
    domain_signal_score: int,
    ai_probability_score: int,
    signal_trust_score: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    response_time_ms: int = 0
):
    """Log a scan to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scans (
            user_id, url, domain_signal_score, ai_probability_score,
            signal_trust_score, input_tokens, output_tokens, cost_usd, response_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, url, domain_signal_score, ai_probability_score,
        signal_trust_score, input_tokens, output_tokens, cost_usd, response_time_ms
    ))

    conn.commit()
    conn.close()


def aggregate_old_data():
    """
    Aggregate data from previous days into daily_aggregates table.
    Called before cleanup to preserve historical stats.
    """
    conn = get_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()

    # Find dates with scans that haven't been aggregated yet (excluding today)
    cursor.execute("""
        SELECT DISTINCT DATE(created_at) as scan_date
        FROM scans
        WHERE DATE(created_at) < ?
        AND DATE(created_at) NOT IN (SELECT agg_date FROM daily_aggregates)
    """, (today,))

    dates_to_aggregate = [row["scan_date"] for row in cursor.fetchall()]

    for agg_date in dates_to_aggregate:
        # Calculate aggregates for this date
        cursor.execute("""
            SELECT
                COUNT(*) as total_scans,
                COUNT(DISTINCT user_id) as unique_users,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cost_usd) as total_cost_usd,
                AVG(response_time_ms) as avg_response_time_ms,
                AVG(signal_trust_score) as avg_trust_score,
                AVG(ai_probability_score) as avg_ai_probability,
                AVG(domain_signal_score) as avg_domain_score
            FROM scans
            WHERE DATE(created_at) = ?
        """, (agg_date,))

        stats = cursor.fetchone()

        if stats and stats["total_scans"] > 0:
            cursor.execute("""
                INSERT INTO daily_aggregates (
                    agg_date, total_scans, unique_users,
                    total_input_tokens, total_output_tokens, total_cost_usd,
                    avg_response_time_ms, avg_trust_score, avg_ai_probability, avg_domain_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agg_date,
                stats["total_scans"],
                stats["unique_users"],
                stats["total_input_tokens"] or 0,
                stats["total_output_tokens"] or 0,
                stats["total_cost_usd"] or 0.0,
                stats["avg_response_time_ms"] or 0.0,
                stats["avg_trust_score"] or 0.0,
                stats["avg_ai_probability"] or 0.0,
                stats["avg_domain_score"] or 0.0,
            ))

    conn.commit()
    conn.close()


def cleanup_old_data(retention_hours: int = 24):
    """
    Aggregate old data, then delete raw records.
    Privacy policy: No user PII stored beyond 24 hours.
    Historical aggregates (no PII) are kept permanently.
    """
    # First, aggregate any old data that hasn't been aggregated
    aggregate_old_data()

    conn = get_connection()
    cursor = conn.cursor()

    # Delete old scans (raw PII data)
    cursor.execute("""
        DELETE FROM scans
        WHERE created_at < DATETIME('now', ? || ' hours')
    """, (f"-{retention_hours}",))

    # Delete old daily limits (older than today)
    cursor.execute("""
        DELETE FROM daily_limits
        WHERE scan_date < DATE('now')
    """)

    # Delete orphaned users (no recent scans)
    cursor.execute("""
        DELETE FROM users
        WHERE user_id NOT IN (SELECT DISTINCT user_id FROM scans)
        AND user_id NOT IN (SELECT DISTINCT user_id FROM daily_limits)
    """)

    conn.commit()
    conn.close()


def get_admin_stats() -> dict:
    """Get aggregate stats for admin dashboard (cumulative + live)."""
    conn = get_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()

    # === CUMULATIVE STATS (from aggregates + today's live data) ===

    # Historical aggregates
    cursor.execute("""
        SELECT
            SUM(total_scans) as scans,
            SUM(unique_users) as users,
            SUM(total_input_tokens) as input_tokens,
            SUM(total_output_tokens) as output_tokens,
            SUM(total_cost_usd) as cost
        FROM daily_aggregates
    """)
    hist = cursor.fetchone()
    hist_scans = hist["scans"] or 0
    hist_users = hist["users"] or 0
    hist_input_tokens = hist["input_tokens"] or 0
    hist_output_tokens = hist["output_tokens"] or 0
    hist_cost = hist["cost"] or 0.0

    # Today's live data
    cursor.execute("""
        SELECT
            COUNT(*) as scans,
            COUNT(DISTINCT user_id) as users,
            SUM(input_tokens) as input_tokens,
            SUM(output_tokens) as output_tokens,
            SUM(cost_usd) as cost,
            AVG(response_time_ms) as avg_response_time,
            AVG(signal_trust_score) as avg_trust,
            AVG(ai_probability_score) as avg_ai_prob,
            AVG(domain_signal_score) as avg_domain
        FROM scans
        WHERE DATE(created_at) = ?
    """, (today,))
    live = cursor.fetchone()
    live_scans = live["scans"] or 0
    live_users = live["users"] or 0
    live_input_tokens = live["input_tokens"] or 0
    live_output_tokens = live["output_tokens"] or 0
    live_cost = live["cost"] or 0.0

    # Cumulative totals
    total_scans = hist_scans + live_scans
    total_input_tokens = hist_input_tokens + live_input_tokens
    total_output_tokens = hist_output_tokens + live_output_tokens
    total_cost = hist_cost + live_cost

    # Total unique users (historical + today, may overlap but approximation is fine)
    cursor.execute("SELECT COUNT(*) as count FROM daily_aggregates")
    num_days = cursor.fetchone()["count"]
    # Estimate: use max of historical daily average or today's users
    avg_hist_users = hist_users / num_days if num_days > 0 else 0
    total_users_estimate = hist_users + live_users  # Upper bound

    # === TODAY'S STATS ===
    scans_today = live_scans
    cost_today = live_cost
    avg_response_time = live["avg_response_time"] or 0
    avg_trust_score = live["avg_trust"] or 0
    avg_ai_probability = live["avg_ai_prob"] or 0

    # === RECENT SCANS (today only, with PII - will be deleted after 24hrs) ===
    cursor.execute("""
        SELECT user_id, url, signal_trust_score, ai_probability_score,
               cost_usd, response_time_ms, created_at
        FROM scans
        ORDER BY created_at DESC
        LIMIT 10
    """)
    recent_scans = [dict(row) for row in cursor.fetchall()]

    # === DAILY STATS (from aggregates + today) ===
    cursor.execute("""
        SELECT
            agg_date as date,
            total_scans as scans,
            total_input_tokens + total_output_tokens as tokens,
            total_cost_usd as cost
        FROM daily_aggregates
        ORDER BY agg_date DESC
        LIMIT 6
    """)
    daily_stats = [dict(row) for row in cursor.fetchall()]

    # Add today's live stats
    if live_scans > 0:
        daily_stats.insert(0, {
            "date": today,
            "scans": live_scans,
            "tokens": live_input_tokens + live_output_tokens,
            "cost": live_cost,
        })

    conn.close()

    return {
        "total_users": total_users_estimate,
        "total_scans": total_scans,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_response_time_ms": round(avg_response_time, 0),
        "scans_today": scans_today,
        "cost_today_usd": round(cost_today, 4),
        "avg_trust_score": round(avg_trust_score, 1),
        "avg_ai_probability": round(avg_ai_probability, 1),
        "avg_domain_score": round(live["avg_domain"] or 0, 1),
        "recent_scans": recent_scans,
        "daily_stats": daily_stats,
    }


# Initialize database on import
init_db()
