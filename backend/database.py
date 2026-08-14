import os
import sqlite3
from datetime import datetime, date, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Turso/libSQL compatibility layer
# --------------------------------------------------------------------------
# libsql returns plain tuples, not sqlite3.Row objects. These wrappers make
# libsql tuples behave like sqlite3.Row so the rest of the code (dict(row),
# row["column"]) works unchanged.


class Row:
    """Wrapper that makes libsql tuples behave like sqlite3.Row."""
    __slots__ = ("_data", "_columns")

    def __init__(self, columns: tuple, values: tuple):
        self._columns = columns
        self._data = dict(zip(columns, values))

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self._data.values())[key]
        return self._data[key]

    def __iter__(self):
        return iter(self._data.items())

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __repr__(self):
        return f"Row({self._data})"


class CursorWrapper:
    """Wraps a libsql cursor to return Row objects instead of tuples."""
    def __init__(self, cursor):
        self._cursor = cursor
        self._columns = None

    def execute(self, sql, params=()):
        result = self._cursor.execute(sql, params)
        if self._cursor.description:
            self._columns = tuple(desc[0] for desc in self._cursor.description)
        return result

    def executescript(self, sql):
        return self._cursor.executescript(sql)

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        if self._columns is None and self._cursor.description:
            self._columns = tuple(desc[0] for desc in self._cursor.description)
        return Row(self._columns, row) if self._columns else row

    def fetchall(self):
        rows = self._cursor.fetchall()
        if self._columns is None and self._cursor.description:
            self._columns = tuple(desc[0] for desc in self._cursor.description)
        if self._columns:
            return [Row(self._columns, row) for row in rows]
        return rows

    def fetchmany(self, size=None):
        rows = self._cursor.fetchmany(size) if size else self._cursor.fetchmany()
        if self._columns is None and self._cursor.description:
            self._columns = tuple(desc[0] for desc in self._cursor.description)
        if self._columns:
            return [Row(self._columns, row) for row in rows]
        return rows

    @property
    def description(self):
        return self._cursor.description

    @property
    def lastrowid(self):
        return self._cursor.lastrowid


class ConnectionWrapper:
    """Wraps a libsql connection to return CursorWrapper."""
    def __init__(self, conn):
        self._conn = conn

    def cursor(self):
        return CursorWrapper(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def close(self):
        return self._conn.close()

    def execute(self, sql, params=()):
        cursor = self.cursor()
        cursor.execute(sql, params)
        return cursor


# --------------------------------------------------------------------------
# Connection setup
# --------------------------------------------------------------------------

# Environment variables for Turso
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

# Local SQLite fallback path (for development)
LOCAL_DB_PATH = Path(__file__).parent / "signalcheck.db"

# Track which backend we're using
_using_turso = False


def get_connection():
    """
    Get a database connection.

    Uses Turso if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set,
    otherwise falls back to local SQLite for development.
    """
    global _using_turso

    if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
        try:
            import libsql_experimental as libsql
            conn = libsql.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
            _using_turso = True
            return ConnectionWrapper(conn)
        except ImportError:
            raise RuntimeError(
                "libsql_experimental not installed. Run: pip install libsql-experimental"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Turso: {e}")
    else:
        # Local SQLite for development
        conn = sqlite3.connect(LOCAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        _using_turso = False
        return conn


def init_db():
    """Initialize database tables."""
    try:
        conn = get_connection()
    except RuntimeError as e:
        # Log the error but don't crash on import - let the first request fail
        # with a clear message instead of breaking the entire module import
        print(f"WARNING: Database initialization failed: {e}")
        print("The API will fail on first request if this is not resolved.")
        return

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
            token_source TEXT DEFAULT 'estimated',
            prompt_version TEXT,
            prompt_hash TEXT,
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
            -- Provenance, carried forward so it survives the 24h purge of `scans`
            zero_token_scans INTEGER DEFAULT 0,
            measured_scans INTEGER DEFAULT 0,
            estimated_scans INTEGER DEFAULT 0,
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

    # Migrations for existing databases. SQLite has no ADD COLUMN IF NOT EXISTS,
    # so each is attempted and the "duplicate column" error is swallowed.
    _migrations = [
        "ALTER TABLE scans ADD COLUMN token_source TEXT DEFAULT 'estimated'",
        "ALTER TABLE daily_aggregates ADD COLUMN zero_token_scans INTEGER DEFAULT 0",
        "ALTER TABLE daily_aggregates ADD COLUMN measured_scans INTEGER DEFAULT 0",
        "ALTER TABLE daily_aggregates ADD COLUMN estimated_scans INTEGER DEFAULT 0",
        # Rows that already exist were all scored by the v1 prompt -- that is what
        # production ran until this commit -- so the DEFAULT is an accurate
        # backfill, not a placeholder. New rows get their value from main.py.
        "ALTER TABLE scans ADD COLUMN prompt_version TEXT DEFAULT 'v1-baseline'",
        "ALTER TABLE scans ADD COLUMN prompt_hash TEXT DEFAULT 'bd8ec8cd'",
    ]
    for stmt in _migrations:
        try:
            cursor.execute(stmt)
        except (sqlite3.OperationalError, Exception):
            pass  # Column already exists

    conn.commit()
    conn.close()

    backend = "Turso" if _using_turso else "local SQLite"
    print(f"Database initialized successfully ({backend})")


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
    # Use UTC date to match SQLite's CURRENT_TIMESTAMP and dashboard stats
    today = datetime.now(timezone.utc).date().isoformat()

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
    # Use UTC date to match SQLite's CURRENT_TIMESTAMP and dashboard stats
    today = datetime.now(timezone.utc).date().isoformat()

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
    response_time_ms: int = 0,
    token_source: str = "estimated",
    prompt_version: str = "unknown",
    prompt_hash: str = "unknown",
):
    """Log a scan to the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scans (
            user_id, url, domain_signal_score, ai_probability_score,
            signal_trust_score, input_tokens, output_tokens, cost_usd, response_time_ms, token_source,
            prompt_version, prompt_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, url, domain_signal_score, ai_probability_score,
        signal_trust_score, input_tokens, output_tokens, cost_usd, response_time_ms, token_source,
        prompt_version, prompt_hash
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
    # Use UTC date to match SQLite's CURRENT_TIMESTAMP (stored in UTC)
    today = datetime.now(timezone.utc).date().isoformat()

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
        # Zero-token rows (failed/instrumented scans) are counted but their
        # token and cost sums are excluded, so per-scan cost is not deflated.
        # The count is carried forward in zero_token_scans so the exclusion
        # survives after `scans` is purged.
        cursor.execute("""
            SELECT
                COUNT(*) as total_scans,
                COUNT(DISTINCT user_id) as unique_users,
                SUM(CASE WHEN input_tokens > 0 OR output_tokens > 0
                         THEN input_tokens ELSE 0 END) as total_input_tokens,
                SUM(CASE WHEN input_tokens > 0 OR output_tokens > 0
                         THEN output_tokens ELSE 0 END) as total_output_tokens,
                SUM(CASE WHEN input_tokens > 0 OR output_tokens > 0
                         THEN cost_usd ELSE 0 END) as total_cost_usd,
                COUNT(CASE WHEN input_tokens = 0 AND output_tokens = 0
                           THEN 1 END) as zero_token_scans,
                COUNT(CASE WHEN token_source = 'measured' THEN 1 END) as measured_scans,
                COUNT(CASE WHEN token_source = 'estimated' OR token_source IS NULL
                           THEN 1 END) as estimated_scans,
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
                    zero_token_scans, measured_scans, estimated_scans,
                    avg_response_time_ms, avg_trust_score, avg_ai_probability, avg_domain_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agg_date,
                stats["total_scans"],
                stats["unique_users"],
                stats["total_input_tokens"] or 0,
                stats["total_output_tokens"] or 0,
                stats["total_cost_usd"] or 0.0,
                stats["zero_token_scans"] or 0,
                stats["measured_scans"] or 0,
                stats["estimated_scans"] or 0,
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
    # Use UTC date to match SQLite's CURRENT_TIMESTAMP (stored in UTC)
    today = datetime.now(timezone.utc).date().isoformat()

    # === CUMULATIVE STATS (from aggregates + today's live data) ===

    # Historical aggregates
    cursor.execute("""
        SELECT
            SUM(total_scans) as scans,
            SUM(unique_users) as users,
            SUM(total_input_tokens) as input_tokens,
            SUM(total_output_tokens) as output_tokens,
            SUM(total_cost_usd) as cost,
            SUM(COALESCE(zero_token_scans, 0)) as zero_token_scans,
            SUM(COALESCE(measured_scans, 0)) as measured_scans,
            SUM(COALESCE(estimated_scans, 0)) as estimated_scans
        FROM daily_aggregates
    """)
    hist = cursor.fetchone()
    hist_scans = hist["scans"] or 0
    hist_users = hist["users"] or 0
    hist_input_tokens = hist["input_tokens"] or 0
    hist_output_tokens = hist["output_tokens"] or 0
    hist_cost = hist["cost"] or 0.0
    hist_zero_token_scans = hist["zero_token_scans"] or 0
    hist_measured_scans = hist["measured_scans"] or 0
    hist_estimated_scans = hist["estimated_scans"] or 0

    # Today's live data (exclude zero-token rows from cost calculations)
    cursor.execute("""
        SELECT
            COUNT(*) as scans,
            COUNT(DISTINCT user_id) as users,
            SUM(CASE WHEN input_tokens > 0 OR output_tokens > 0 THEN input_tokens ELSE 0 END) as input_tokens,
            SUM(CASE WHEN input_tokens > 0 OR output_tokens > 0 THEN output_tokens ELSE 0 END) as output_tokens,
            SUM(CASE WHEN input_tokens > 0 OR output_tokens > 0 THEN cost_usd ELSE 0 END) as cost,
            COUNT(CASE WHEN input_tokens = 0 AND output_tokens = 0 THEN 1 END) as zero_token_scans,
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
    live_zero_token_scans = live["zero_token_scans"] or 0

    # Provenance for TODAY only. `scans` is purged after 24h, so anything older
    # has to come from daily_aggregates -- see hist_* above.
    cursor.execute("""
        SELECT
            COUNT(CASE WHEN token_source = 'measured' THEN 1 END) as measured,
            COUNT(CASE WHEN token_source = 'estimated' OR token_source IS NULL THEN 1 END) as estimated
        FROM scans
        WHERE DATE(created_at) = ?
    """, (today,))
    provenance = cursor.fetchone()
    measured_scans = hist_measured_scans + (provenance["measured"] or 0)
    estimated_scans = hist_estimated_scans + (provenance["estimated"] or 0)

    # Cumulative totals
    total_scans = hist_scans + live_scans
    total_input_tokens = hist_input_tokens + live_input_tokens
    total_output_tokens = hist_output_tokens + live_output_tokens
    total_cost = hist_cost + live_cost
    total_zero_token_scans = hist_zero_token_scans + live_zero_token_scans

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
               input_tokens, output_tokens, response_time_ms, created_at,
               COALESCE(token_source, 'estimated') as token_source
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
        "zero_token_scans": total_zero_token_scans,
        "measured_scans": measured_scans,
        "estimated_scans": estimated_scans,
        # Denominator for cost-per-scan: excludes rows with no token data, both
        # today's and those already rolled into daily_aggregates.
        "valid_scans_for_cost": max(total_scans - total_zero_token_scans, 0),
    }


# Initialize database on import
init_db()
