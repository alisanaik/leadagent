import sqlite3
import json
from typing import Optional

DB_PATH = "cache.db"


def _get_connection() -> sqlite3.Connection:
    """Create (or open) the SQLite database file."""
    conn = sqlite3.connect(DB_PATH)
    # Create the table if it doesn't exist yet
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_cache (
            domain TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def get_cached_result(domain: str) -> Optional[dict]:
    """Return the cached result for a domain, or None if not cached."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT result_json FROM domain_cache WHERE domain = ?",
            (domain,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    finally:
        conn.close()


def save_result(domain: str, result_dict: dict) -> None:
    """Store or update the cached result for a domain."""
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO domain_cache (domain, result_json) VALUES (?, ?)",
            (domain, json.dumps(result_dict))
        )
        conn.commit()
    finally:
        conn.close()


def clear_cache() -> None:
    """Wipe the cache. Useful during development when iterating on prompts."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM domain_cache")
        conn.commit()
        print("[Cache] Cache cleared.")
    finally:
        conn.close()
