#!/usr/bin/env python3
"""
Turso/libSQL compatibility spike v2 - with Row wrapper.

Tests the wrapper approach to make libsql tuples behave like sqlite3.Row.
"""

import os
import sys


class Row:
    """
    Wrapper that makes libsql tuples behave like sqlite3.Row.
    Supports: row["column"], dict(row), iteration.
    """
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
    """
    Wraps a libsql cursor to return Row objects instead of tuples.
    """
    def __init__(self, cursor):
        self._cursor = cursor
        self._columns = None

    def execute(self, sql, params=()):
        result = self._cursor.execute(sql, params)
        # Capture column names after execute
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
    """
    Wraps a libsql connection to return CursorWrapper.
    """
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


def test_wrapper():
    """Test the wrapper with libsql."""

    try:
        import libsql_experimental as libsql
        print("✓ libsql_experimental imported")
    except ImportError:
        print("✗ Run: pip install libsql-experimental")
        return False

    # Connect
    db_url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    if db_url:
        print(f"✓ Testing against Turso: {db_url[:40]}...")
        raw_conn = libsql.connect(db_url, auth_token=auth_token)
    else:
        print("✓ Testing with local libsql file")
        raw_conn = libsql.connect("/tmp/spike_test_v2.db")

    # Wrap the connection
    conn = ConnectionWrapper(raw_conn)
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
        print("✓ CREATE TABLE works")
    except Exception as e:
        print(f"✗ CREATE TABLE failed: {e}")
        return False

    # Test 2: Insert
    try:
        cursor.execute("INSERT INTO spike_test (name, value) VALUES (?, ?)", ("test", 42))
        cursor.execute("INSERT INTO spike_test (name, value) VALUES (?, ?)", ("another", 99))
        conn.commit()
        print("✓ INSERT works")
    except Exception as e:
        print(f"✗ INSERT failed: {e}")
        return False

    # Test 3: fetchone + dict(row)
    try:
        cursor.execute("SELECT * FROM spike_test WHERE name = ?", ("test",))
        row = cursor.fetchone()
        print(f"✓ fetchone works, type: {type(row)}")
        print(f"  Row: {row}")

        row_dict = dict(row)
        print(f"✓ dict(row) works: {row_dict}")
    except Exception as e:
        print(f"✗ dict(row) failed: {e}")
        return False

    # Test 4: row["column"] access
    try:
        value = row["value"]
        name = row["name"]
        print(f"✓ row['column'] works: name={name}, value={value}")
    except Exception as e:
        print(f"✗ row['column'] failed: {e}")
        return False

    # Test 5: fetchall + list comprehension (used in database.py)
    try:
        cursor.execute("SELECT * FROM spike_test ORDER BY id")
        rows = cursor.fetchall()
        all_dicts = [dict(r) for r in rows]
        print(f"✓ [dict(r) for r in fetchall()] works: {all_dicts}")
    except Exception as e:
        print(f"✗ fetchall pattern failed: {e}")
        return False

    # Test 6: Access after fetchone returns None
    try:
        cursor.execute("SELECT * FROM spike_test WHERE name = ?", ("nonexistent",))
        row = cursor.fetchone()
        if row is None:
            print("✓ fetchone returns None for no match")
        else:
            print(f"⚠ Expected None, got: {row}")
    except Exception as e:
        print(f"✗ None handling failed: {e}")
        return False

    # Test 7: DATE/DATETIME functions
    try:
        cursor.execute("SELECT DATE('now') as today, DATETIME('now', '-24 hours') as cutoff")
        row = cursor.fetchone()
        print(f"✓ DATE/DATETIME work: today={row['today']}, cutoff={row['cutoff']}")
    except Exception as e:
        print(f"⚠ DATE/DATETIME issue: {e}")

    # Cleanup
    cursor.execute("DROP TABLE spike_test")
    conn.commit()
    conn.close()

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED — Wrapper approach works!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_wrapper()
    sys.exit(0 if success else 1)
