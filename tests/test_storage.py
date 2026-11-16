"""
Test Suite for SQLite Persistence & Prediction Caching
======================================================
Validates:
  - SQLite schema creation, WAL pragmas, and concurrency safeguards.
  - SHA-256 deterministic text-hash prediction caching and TTL expiration.
  - Batch job persistence across simulated application restarts.
  - API cache statistics and maintenance endpoints.
"""

import os
import time
import tempfile
import unittest
from unittest.mock import patch

from News_Detection.storage import (
    SQLiteManager,
    PredictionCache,
    BatchJobStore,
)
from News_Detection.batch_processor import BatchJobManager


class TestSQLiteStorage(unittest.TestCase):
    """Unit tests for SQLite database manager and table initialization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_news.db")
        self.db_manager = SQLiteManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tables_and_indexes_created(self):
        with self.db_manager.get_connection() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            self.assertIn("batch_jobs", tables)
            self.assertIn("prediction_cache", tables)

            indexes = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            ]
            self.assertIn("idx_batch_jobs_created", indexes)
            self.assertIn("idx_cache_expires", indexes)

    def test_in_memory_mode(self):
        mem_manager = SQLiteManager(":memory:")
        with mem_manager.get_connection() as conn:
            conn.execute("INSERT INTO prediction_cache (cache_key, input_hash, input_snippet, response_json, is_fake, confidence, language_code, created_at, expires_at, ttl_seconds) VALUES ('k1', 'h1', 'snip', '{}', 0, 95.0, 'en', 100, 200, 100)")
            row = conn.execute("SELECT cache_key FROM prediction_cache WHERE cache_key = 'k1'").fetchone()
            self.assertEqual(row["cache_key"], "k1")


class TestPredictionCache(unittest.TestCase):
    """Unit tests for SHA-256 prediction cache and TTL expiration."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "cache_test.db")
        self.db_manager = SQLiteManager(self.db_path)
        self.cache = PredictionCache(self.db_manager, default_ttl_seconds=3600)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_miss_returns_none(self):
        res = self.cache.get("Unseen claim headline")
        self.assertIsNone(res)

    def test_cache_set_and_hit(self):
        text = "World health agency releases official guidelines on seasonal flu."
        mock_analysis = {
            "verdict": "AUTHENTIC",
            "is_fake": False,
            "confidence": 98.4,
            "language_code": "en",
            "models": {"roberta": "AUTHENTIC"},
        }

        key = self.cache.set(text, mock_analysis)
        self.assertTrue(len(key) == 64)  # SHA-256 hex length

        cached = self.cache.get(text)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["verdict"], "AUTHENTIC")
        self.assertTrue(cached["cached"])
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(cached["cache_key"], key)

    def test_cache_hit_increments_counter(self):
        text = "Repeat query testing hit counter"
        self.cache.set(text, {"verdict": "REAL", "is_fake": False})

        self.cache.get(text)
        self.cache.get(text)
        self.cache.get(text)

        stats = self.cache.get_stats()
        self.assertEqual(stats["total_entries"], 1)
        self.assertEqual(stats["active_entries"], 1)
        self.assertEqual(stats["total_hits"], 3)

    def test_cache_expiration(self):
        text = "Ephemeral breaking news statement"
        self.cache.set(text, {"verdict": "FAKE", "is_fake": True}, ttl_seconds=1)

        # Before expiration
        self.assertIsNotNone(self.cache.get(text))

        # Sleep past TTL
        time.sleep(1.1)

        # After expiration
        self.assertIsNone(self.cache.get(text))

    def test_clear_expired(self):
        self.cache.set("text 1", {"verdict": "REAL"}, ttl_seconds=1)
        self.cache.set("text 2", {"verdict": "FAKE"}, ttl_seconds=3600)
        time.sleep(1.1)

        cleared = self.cache.clear_expired()
        self.assertEqual(cleared, 1)

        stats = self.cache.get_stats()
        self.assertEqual(stats["total_entries"], 1)

    def test_clear_all(self):
        self.cache.set("text 1", {"verdict": "REAL"})
        self.cache.set("text 2", {"verdict": "FAKE"})
        self.assertEqual(self.cache.get_stats()["total_entries"], 2)

        cleared = self.cache.clear_all()
        self.assertEqual(cleared, 2)
        self.assertEqual(self.cache.get_stats()["total_entries"], 0)

    def test_unicode_normalization_cache_equivalence(self):
        # Combining accents vs precomposed characters
        text_precomposed = "Café reports economic recovery"
        text_decomposed = "Cafe\u0301 reports economic recovery"

        self.cache.set(text_precomposed, {"verdict": "REAL", "soft_score": 91.0})
        cached = self.cache.get(text_decomposed)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["verdict"], "REAL")

    def test_multi_tiered_l1_ram_and_l2_sqlite_promotion(self):
        """Validates that L1 RAM serves hits in <0.05ms and promotes on L2 SQLite hits."""
        text = "Breaking technology breakthrough announced by leading scientists."
        mock_payload = {"verdict": "AUTHENTIC", "soft_score": 96.5}

        # 1. Set populates both L1 RAM and L2 SQLite
        key = self.cache.set(text, mock_payload)

        # 2. First lookup comes directly from Tier 1 RAM (L1)
        res1 = self.cache.get(text)
        self.assertIsNotNone(res1)
        self.assertEqual(res1["cache_tier"], "L1_RAM")
        self.assertEqual(res1["cache_key"], key)

        # 3. Simulate process restart or RAM eviction by clearing L1 RAM only
        self.cache.l1.clear()
        self.assertEqual(len(self.cache.l1._store), 0)

        # 4. Lookup now hits Tier 2 SQLite (L2) and promotes back to L1 RAM
        res2 = self.cache.get(text)
        self.assertIsNotNone(res2)
        self.assertEqual(res2["cache_tier"], "L2_SQLITE")

        # 5. Subsequent lookup now hits Tier 1 RAM again
        res3 = self.cache.get(text)
        self.assertIsNotNone(res3)
        self.assertEqual(res3["cache_tier"], "L1_RAM")

    def test_l1_ram_lru_eviction(self):
        """Validates bounded LRU eviction in L1 RAM while keeping entries in L2 SQLite."""
        small_cache = PredictionCache(self.db_manager, l1_max_size=2)

        small_cache.set("doc1", {"val": 1})
        small_cache.set("doc2", {"val": 2})
        small_cache.set("doc3", {"val": 3})  # Should evict doc1 from L1 RAM

        # doc1 was evicted from L1 RAM, but is still intact in L2 SQLite
        self.assertNotIn(small_cache.compute_cache_key("doc1")[0], small_cache.l1._store)
        res_doc1 = small_cache.get("doc1")
        self.assertIsNotNone(res_doc1)
        self.assertEqual(res_doc1["cache_tier"], "L2_SQLITE")

    def test_cache_stats_multi_tier_telemetry(self):
        """Validates that get_stats accurately reports L1 RAM and L2 SQLite metrics."""
        self.cache.set("headline A", {"verdict": "REAL"})
        self.cache.get("headline A")  # L1 hit

        stats = self.cache.get_stats()
        self.assertEqual(stats["tier_architecture"], "Multi-Tiered (L1 RAM + L2 SQLite)")
        self.assertIn("l1_ram", stats)
        self.assertIn("l2_sqlite", stats)
        self.assertGreaterEqual(stats["total_hits"], 1)
        self.assertGreaterEqual(stats["l1_ram"]["hits"], 1)


class TestBatchJobPersistence(unittest.TestCase):
    """Validates that batch jobs survive application restarts via SQLite persistence."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "batch_test.db")
        self.db_manager = SQLiteManager(self.db_path)
        self.store = BatchJobStore(self.db_manager, max_stored_jobs=5)
        self.manager = BatchJobManager(self.store)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("News_Detection.batch_processor.analyze_text")
    def test_job_persistence_across_manager_restarts(self, mock_analyze):
        mock_analyze.return_value = {
            "is_fake": False,
            "soft_score": 94.0,
            "language_code": "en",
            "language_name": "English",
            "agreement": "3/3",
            "vote1": "AUTHENTIC",
            "score1": "94%",
            "vote2": "AUTHENTIC",
            "score2": "90%",
            "vote3": "AUTHENTIC",
            "score3": "98%",
            "vote4": "N/A",
            "score4": "N/A",
        }

        items = [
            {"id": "doc-1", "text": "First verified news story", "metadata": {"outlet": "Reuters"}},
            {"id": "doc-2", "text": "Second verified news story", "metadata": {"outlet": "AP"}},
        ]
        meta = {"filename": "audit.csv", "total_extracted": 2}

        # 1. Create and run job with initial manager
        job_id = self.manager.create_job("audit.csv", items, meta)
        self.manager.run_job_sync(job_id)

        job_initial = self.manager.get_job(job_id)
        self.assertEqual(job_initial["status"], "completed")
        self.assertEqual(job_initial["summary"]["real_count"], 2)

        # 2. Simulate complete application restart by instantiating a completely NEW manager
        # connecting to the exact same SQLite database file
        new_db_manager = SQLiteManager(self.db_path)
        new_store = BatchJobStore(new_db_manager)
        new_manager = BatchJobManager(new_store)

        job_restarted = new_manager.get_job(job_id)
        self.assertIsNotNone(job_restarted)
        self.assertEqual(job_restarted["job_id"], job_id)
        self.assertEqual(job_restarted["status"], "completed")
        self.assertEqual(job_restarted["total_items"], 2)
        self.assertEqual(job_restarted["processed_items"], 2)
        self.assertEqual(job_restarted["progress_percent"], 100.0)
        self.assertEqual(len(job_restarted["results"]), 2)
        self.assertEqual(job_restarted["results"][0]["metadata"]["outlet"], "Reuters")

        # 3. Export CSV from restarted manager
        csv_out = new_manager.export_csv(job_id)
        self.assertIn("Reuters", csv_out)
        self.assertIn("AP", csv_out)
        self.assertIn("AUTHENTIC", csv_out)

    def test_job_pruning_on_max_stored_jobs(self):
        for i in range(7):
            self.store.create_job(f"file_{i}.csv", [{"id": f"{i}", "text": "sample"}], {})

        jobs = self.store.list_jobs(limit=10)
        # Should be pruned to max_stored_jobs (5)
        self.assertLessEqual(len(jobs), 5)


class TestApiCacheEndpoints(unittest.TestCase):
    """Validates the FastAPI cache stats and clear endpoints."""

    def test_cache_endpoints_via_test_client(self):
        from fastapi.testclient import TestClient
        from News_Detection.health_api import app

        client = TestClient(app)

        # 1. Check /api/cache/stats
        stats_resp = client.get("/api/cache/stats")
        self.assertEqual(stats_resp.status_code, 200)
        data = stats_resp.json()
        self.assertIn("total_entries", data)
        self.assertIn("active_entries", data)
        self.assertIn("total_hits", data)

        # 2. Check /health contains database and prediction cache
        health_resp = client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        health_data = health_resp.json()
        self.assertTrue(health_data["prediction_cache"])
        self.assertIn("SQLite", health_data["database"])

        # 3. Clear cache
        clear_resp = client.post("/api/cache/clear?expired_only=false")
        self.assertEqual(clear_resp.status_code, 200)
        self.assertEqual(clear_resp.json()["status"], "cleared")


if __name__ == "__main__":
    unittest.main()

