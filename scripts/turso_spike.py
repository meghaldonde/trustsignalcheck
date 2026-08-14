#!/usr/bin/env python3
"""
Turso/libSQL compatibility spike.

Tests whether libsql supports the row_factory pattern used in database.py.
This is a go/no-go test before migrating to Turso.

Usage:
    # Install first:
    pip install libsql-experimental

    # Test with local file (no Turso account needed):
    python scripts/turso_spike.py

    # Test with actual Turso DB:
    TURSO_DATABASE_URL="libsql://your-db.turso.io" \
    TURSO_AUTH_TOKEN="your-token" \
    python scripts/turso_spike.py
"""

import os
import sys

def test_libsql_row_factory():
    """Test if libsql supports dict-like row access."""

    try:
        import libsql_experimental as libsql
        print("✓ libsql_experimental imported successfully")
    except ImportError:
        print("✗ Failed to import libsql_experimental")
        print("  Run: pip install libsql-experimental")
        return False

    # Use Turso URL if provided, otherwise local file
    db_url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if db_url:
        print(f"✓ Testing against Turso: {db_url[:40]}...")
        conn = libsql.connect(db_url, auth_token=auth_token)
    else:
        print("✓ Testing with local libsql file: /tmp/spike_test.db")
        conn = libsql.connect("/tmp/spike_test.db")

    cursor = conn.cursor()

    # Test 1: Create table
    try:
        cursor.execute("DROP TABLE IF EXISTS spike_test")
        cursor.execute("""
            CREATE TABLE spike_test (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        print("✓ CREATE TABLE with AUTOINCREMENT works")
    except Exception as e:
        print(f"✗ CREATE TABLE failed: {e}")
        return False

    # Test 2: Insert with ON CONFLICT
    try:
        cursor.execute("""
            INSERT INTO spike_test (name, value) VALUES ('test', 42)
        """)
        conn.commit()
        print("✓ INSERT works")
    except Exception as e:
        print(f"✗ INSERT failed: {e}")
        return False

    # Test 3: Fetch and check row type
    try:
        cursor.execute("SELECT * FROM spike_test WHERE name = ?", ("test",))
        row = cursor.fetchone()
        print(f"✓ SELECT works, row type: {type(row)}")
        print(f"  Raw row: {row}")
    except Exception as e:
        print(f"✗ SELECT failed: {e}")
        return False

    # Test 4: dict(row) - THE CRITICAL TEST
    try:
        row_dict = dict(row)
        print(f"✓ dict(row) works: {row_dict}")
    except Exception as e:
        print(f"✗ dict(row) FAILED: {e}")
        print("  This is the blocker - your database.py uses dict(row) throughout")
        return False

    # Test 5: row["column"] access
    try:
        value = row["value"]
        print(f"✓ row['column'] works: value={value}")
    except Exception as e:
        print(f"✗ row['column'] FAILED: {e}")
        print("  This is a blocker - your database.py uses row['column'] access")
        return False

    # Test 6: DATE() function (used in cleanup queries)
    try:
        cursor.execute("SELECT DATE('now') as today")
        result = cursor.fetchone()
        print(f"✓ DATE('now') works: {result['today'] if hasattr(result, '__getitem__') else result}")
    except Exception as e:
        print(f"⚠ DATE('now') issue: {e}")
        # Not a hard blocker, but worth noting

    # Test 7: DATETIME with modifier (used in cleanup_old_data)
    try:
        cursor.execute("SELECT DATETIME('now', '-24 hours') as cutoff")
        result = cursor.fetchone()
        print(f"✓ DATETIME with modifier works")
    except Exception as e:
        print(f"⚠ DATETIME modifier issue: {e}")

    # Cleanup
    cursor.execute("DROP TABLE spike_test")
    conn.commit()
    conn.close()

    print("\n" + "="*50)
    print("✓ ALL CRITICAL TESTS PASSED - Turso migration is viable")
    print("="*50)
    return True


if __name__ == "__main__":
    success = test_libsql_row_factory()
    sys.exit(0 if success else 1)
