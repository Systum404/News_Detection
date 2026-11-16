"""
Persistent SQLite Storage & Neural Prediction Cache
===================================================
High-performance local SQLite storage module providing:
  1. Durable batch processing job state management (survives process restarts).
  2. SHA-256 text-hash prediction caching with configurable TTL.
  3. WAL (Write-Ahead Logging) concurrency optimization for thread-safe operations.
"""

import os
import json
import time
import uuid
import sqlite3
import logging
import hashlib
import unicodedata
import threading
from contextlib import contextmanager
from datetime import datetime
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Generator

logger = logging.getLogger("news_detection.storage")

# Default database location: <repo_root>/data/news_detection.db
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.getenv(
    "NEWS_DETECTION_DB_PATH",
    os.path.join(REPO_ROOT, "data", "news_detection.db"),
)
DEFAULT_CACHE_TTL_SECONDS = int(os.getenv("PREDICTION_CACHE_TTL_SECONDS", "86400"))  # 24 Hours
MAX_STORED_BATCH_JOBS = int(os.getenv("MAX_STORED_JOBS", "50"))


class SQLiteManager:
    """
    Manages SQLite database connections with WAL mode and concurrency timeouts.
    Supports file-based SQLite databases with thread-isolated connections and
    in-memory databases for isolated unit testing.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._is_memory = self.db_path == ":memory:"
        self._mem_conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

        if not self._is_memory:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        else:
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row

        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides a managed SQLite connection with automatic commit and rollback."""
        if self._is_memory:
            with self._lock:
                try:
                    yield self._mem_conn  # type: ignore
                    self._mem_conn.commit()  # type: ignore
                except Exception:
                    self._mem_conn.rollback()  # type: ignore
                    raise
        else:
            conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def init_db(self):
        """Initializes relational tables and performance indexes if they do not exist."""
        with self.get_connection() as conn:
            # 1. Batch Processing Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_items INTEGER NOT NULL,
                    processed_items INTEGER NOT NULL DEFAULT 0,
                    progress_percent REAL NOT NULL DEFAULT 0.0,
                    error TEXT,
                    enable_grounding INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    items_to_process_json TEXT,
                    results_json TEXT NOT NULL DEFAULT '[]',
                    summary_json TEXT NOT NULL DEFAULT '{}'
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_jobs_created ON batch_jobs(created_at DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);")

            # 2. Prediction Cache Table (SHA-256 hashed queries)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    cache_key TEXT PRIMARY KEY,
                    input_hash TEXT NOT NULL,
                    input_snippet TEXT NOT NULL,
                    categories TEXT,
                    enable_grounding INTEGER NOT NULL DEFAULT 0,
                    response_json TEXT NOT NULL,
                    is_fake INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    language_code TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0,
                    last_hit_at REAL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_expires ON prediction_cache(expires_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_input_hash ON prediction_cache(input_hash);")


class InRamL1Cache:
    """
    High-throughput, thread-safe in-memory LRU cache (Tier 1).
    Serves repeated queries in <0.001ms without disk I/O, SQLite file locks, or JSON parsing.
    """

    def __init__(self, maxsize: int = 2048):
        self.maxsize = maxsize
        self._store: OrderedDict[str, Tuple[Dict[str, Any], float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str, now: float) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            val, expires_at = self._store[key]
            if expires_at <= now:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return dict(val)

    def set(self, key: str, value: Dict[str, Any], expires_at: float) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (dict(value), expires_at)
            if len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear_expired(self, now: float) -> int:
        with self._lock:
            expired_keys = [k for k, (_, exp) in self._store.items() if exp <= now]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            ratio = round((self.hits / total * 100.0), 1) if total > 0 else 0.0
            return {
                "max_size": self.maxsize,
                "current_size": len(self._store),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio_pct": ratio,
            }


class PredictionCache:
    """
    Multi-tiered text-hash prediction cache:
      - Tier 1 (L1): In-Memory RAM LRU cache (<0.001ms latency, 0 disk I/O).
      - Tier 2 (L2): Durable SQLite database in WAL mode (~0.5ms latency, restart-persistent).
    """

    def __init__(
        self,
        db_manager: SQLiteManager,
        default_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        l1_max_size: int = 2048,
    ):
        self.db = db_manager
        self.default_ttl = default_ttl_seconds
        self.l1 = InRamL1Cache(maxsize=l1_max_size)

    @staticmethod
    def compute_cache_key(
        text: str,
        categories: Optional[str] = None,
        enable_grounding: bool = False,
    ) -> Tuple[str, str]:
        """
        Computes deterministic SHA-256 digests for normalized article text and cache query.
        Returns: (cache_key, raw_text_hash)
        """
        normalized_text = unicodedata.normalize("NFKC", text.strip())
        raw_text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        compound = f"{raw_text_hash}__cat:{categories or ''}__ground:{int(bool(enable_grounding))}"
        cache_key = hashlib.sha256(compound.encode("utf-8")).hexdigest()
        return cache_key, raw_text_hash

    def get(
        self,
        text: str,
        categories: Optional[str] = None,
        enable_grounding: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached analysis result if it exists and has not expired.
        Checks Tier 1 RAM first; falls back to Tier 2 SQLite and promotes on hit.
        """
        cache_key, raw_text_hash = self.compute_cache_key(text, categories, enable_grounding)
        now = time.time()

        # 1. Tier 1: Check In-Memory RAM Cache (L1)
        l1_item = self.l1.get(cache_key, now)
        if l1_item is not None:
            l1_item["cached"] = True
            l1_item["cache_hit"] = True
            l1_item["cache_tier"] = "L1_RAM"
            l1_item["cache_key"] = cache_key
            l1_item.setdefault("claim_sha256", raw_text_hash)
            return l1_item

        # 2. Tier 2: Check SQLite Disk Cache (L2)
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT response_json, expires_at, hits FROM prediction_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()

                if not row:
                    return None

                if row["expires_at"] <= now:
                    # Expired entry - clean up lazily
                    conn.execute("DELETE FROM prediction_cache WHERE cache_key = ?", (cache_key,))
                    return None

                # Record cache hit in SQLite
                conn.execute(
                    "UPDATE prediction_cache SET hits = hits + 1, last_hit_at = ? WHERE cache_key = ?",
                    (now, cache_key),
                )

                cached_result: Dict[str, Any] = json.loads(row["response_json"])

                # Promote to Tier 1 RAM Cache (L1)
                self.l1.set(cache_key, cached_result, row["expires_at"])

                cached_result["cached"] = True
                cached_result["cache_hit"] = True
                cached_result["cache_tier"] = "L2_SQLITE"
                cached_result["cache_key"] = cache_key
                cached_result.setdefault("claim_sha256", raw_text_hash)
                return cached_result

        except Exception as exc:
            logger.warning(f"Cache lookup failed for key {cache_key}: {exc}")
            return None

    def set(
        self,
        text: str,
        result: Dict[str, Any],
        categories: Optional[str] = None,
        enable_grounding: bool = False,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Stores a prediction result in both Tier 1 RAM and Tier 2 SQLite caches."""
        cache_key, raw_hash = self.compute_cache_key(text, categories, enable_grounding)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()
        expires_at = now + ttl

        # Strip volatile cache-status flags before persistence
        clean_result = dict(result)
        clean_result.pop("cached", None)
        clean_result.pop("cache_hit", None)
        clean_result.pop("cache_key", None)
        clean_result.pop("cache_tier", None)

        # 1. Populate Tier 1 RAM Cache (L1)
        self.l1.set(cache_key, clean_result, expires_at)

        # 2. Populate Tier 2 SQLite Disk Cache (L2)
        snippet = text.strip()[:150]
        is_fake = 1 if clean_result.get("is_fake") else 0
        confidence = float(clean_result.get("soft_score") or clean_result.get("confidence") or 0.0)
        language_code = str(clean_result.get("language_code") or clean_result.get("lang") or "en")

        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO prediction_cache (
                        cache_key, input_hash, input_snippet, categories,
                        enable_grounding, response_json, is_fake, confidence,
                        language_code, created_at, expires_at, ttl_seconds, hits, last_hit_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                    """,
                    (
                        cache_key,
                        raw_hash,
                        snippet,
                        categories,
                        1 if enable_grounding else 0,
                        json.dumps(clean_result),
                        is_fake,
                        confidence,
                        language_code,
                        now,
                        expires_at,
                        ttl,
                    ),
                )
            return cache_key
        except Exception as exc:
            logger.error(f"Failed storing prediction cache for key {cache_key}: {exc}")
            return cache_key

    def clear_expired(self) -> int:
        """Removes expired entries from both Tier 1 RAM and Tier 2 SQLite stores."""
        now = time.time()
        self.l1.clear_expired(now)
        with self.db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM prediction_cache WHERE expires_at <= ?", (now,))
            return cursor.rowcount

    def clear_all(self) -> int:
        """Flushes both Tier 1 RAM and Tier 2 SQLite prediction caches."""
        self.l1.clear()
        with self.db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM prediction_cache")
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """Returns diagnostic metrics regarding cache volume, hits, and expired rows."""
        now = time.time()
        with self.db.get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM prediction_cache").fetchone()[0]
            active = conn.execute("SELECT COUNT(*) FROM prediction_cache WHERE expires_at > ?", (now,)).fetchone()[0]
            sqlite_hits = conn.execute("SELECT COALESCE(SUM(hits), 0) FROM prediction_cache").fetchone()[0]

        l1_stats = self.l1.get_stats()
        total_hits = l1_stats["hits"] + sqlite_hits

        return {
            "tier_architecture": "Multi-Tiered (L1 RAM + L2 SQLite)",
            "total_entries": total,
            "active_entries": active,
            "expired_entries": total - active,
            "total_hits": total_hits,
            "l1_ram": l1_stats,
            "l2_sqlite": {
                "active_entries": active,
                "hits": sqlite_hits,
            },
            "default_ttl_seconds": self.default_ttl,
            "db_path": self.db.db_path,
        }


class BatchJobStore:
    """
    Durable batch job persistence backed by SQLite.
    Guarantees that job metadata, row-level evaluations, and summary metrics
    survive service restarts.
    """

    def __init__(self, db_manager: SQLiteManager, max_stored_jobs: int = MAX_STORED_BATCH_JOBS):
        self.db = db_manager
        self.max_stored_jobs = max_stored_jobs

    def create_job(
        self,
        filename: str,
        items: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        enable_grounding: bool = False,
    ) -> str:
        """Registers a new batch job in SQLite and returns its unique job_id."""
        job_id = uuid.uuid4().hex[:12]
        created_at = datetime.now().isoformat()
        total_items = len(items)

        initial_summary = {
            "total": total_items,
            "real_count": 0,
            "fake_count": 0,
            "real_pct": 0.0,
            "fake_pct": 0.0,
            "avg_confidence": 0.0,
            "languages": {},
        }

        with self.db.get_connection() as conn:
            # Evict oldest jobs beyond retention limit
            conn.execute(
                f"""
                DELETE FROM batch_jobs WHERE job_id NOT IN (
                    SELECT job_id FROM batch_jobs ORDER BY created_at DESC LIMIT {self.max_stored_jobs - 1}
                )
                """
            )

            conn.execute(
                """
                INSERT INTO batch_jobs (
                    job_id, filename, status, created_at, completed_at,
                    total_items, processed_items, progress_percent, error,
                    enable_grounding, metadata_json, items_to_process_json,
                    results_json, summary_json
                ) VALUES (?, ?, ?, ?, NULL, ?, 0, 0.0, NULL, ?, ?, ?, '[]', ?)
                """,
                (
                    job_id,
                    filename,
                    "queued",
                    created_at,
                    total_items,
                    1 if enable_grounding else 0,
                    json.dumps(metadata),
                    json.dumps(items),
                    json.dumps(initial_summary),
                ),
            )

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a job by ID, returning metadata, progress, summary, and row-level results.
        Omits internal unprocessed items payload.
        """
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT job_id, filename, status, created_at, completed_at,
                       total_items, processed_items, progress_percent, error,
                       enable_grounding, metadata_json, results_json, summary_json
                FROM batch_jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()

            if not row:
                return None

            return {
                "job_id": row["job_id"],
                "filename": row["filename"],
                "status": row["status"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "total_items": row["total_items"],
                "processed_items": row["processed_items"],
                "progress_percent": float(row["progress_percent"]),
                "error": row["error"],
                "enable_grounding": bool(row["enable_grounding"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "results": json.loads(row["results_json"] or "[]"),
                "summary": json.loads(row["summary_json"] or "{}"),
            }

    def get_items_to_process(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieves raw input items for job processing."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT items_to_process_json FROM batch_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row or not row["items_to_process_json"]:
                return []
            return json.loads(row["items_to_process_json"])

    def update_progress(self, job_id: str, processed_items: int, total_items: int):
        """Updates processed count and percentage."""
        progress_pct = round((processed_items / max(1, total_items)) * 100, 1)
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE batch_jobs
                SET processed_items = ?, progress_percent = ?
                WHERE job_id = ?
                """,
                (processed_items, progress_pct, job_id),
            )

    def set_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Sets status string (e.g. 'processing', 'failed')."""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE batch_jobs SET status = ?, error = ? WHERE job_id = ?",
                (status, error, job_id),
            )

    def complete_job(
        self,
        job_id: str,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ):
        """Marks a job as completed and stores the finalized results and analytics."""
        completed_at = datetime.now().isoformat()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE batch_jobs
                SET status = 'completed',
                    completed_at = ?,
                    processed_items = total_items,
                    progress_percent = 100.0,
                    results_json = ?,
                    summary_json = ?,
                    items_to_process_json = NULL
                WHERE job_id = ?
                """,
                (completed_at, json.dumps(results), json.dumps(summary), job_id),
            )

    def fail_job(self, job_id: str, error_message: str):
        """Marks a job as failed with an error description."""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE batch_jobs SET status = 'failed', error = ? WHERE job_id = ?",
                (error_message, job_id),
            )

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns lightweight overview of recent jobs for UI dashboards."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT job_id, filename, status, created_at, completed_at,
                       total_items, processed_items, progress_percent, summary_json
                FROM batch_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            return [
                {
                    "job_id": r["job_id"],
                    "filename": r["filename"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "completed_at": r["completed_at"],
                    "total_items": r["total_items"],
                    "processed_items": r["processed_items"],
                    "progress_percent": float(r["progress_percent"]),
                    "summary": json.loads(r["summary_json"] or "{}"),
                }
                for r in rows
            ]

    def delete_job(self, job_id: str) -> bool:
        """Deletes a job from the database."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("DELETE FROM batch_jobs WHERE job_id = ?", (job_id,))
            return cursor.rowcount > 0


# --- GLOBAL SINGLETON INSTANCES ---
db_manager = SQLiteManager()
prediction_cache = PredictionCache(db_manager)
batch_job_store = BatchJobStore(db_manager)

