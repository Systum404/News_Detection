import unittest
from unittest.mock import patch

from News_Detection.batch_processor import (
    parse_batch_file,
    detect_csv_text_column,
    BatchJobManager,
    BatchProcessingError,
)


class TestBatchProcessor(unittest.TestCase):
    """Test suite for batch document ingestion, parsing, and execution."""

    def test_parse_csv_with_explicit_text_column(self):
        csv_data = b"id,headline,author\n1,Space telescope discovers new galaxy,Dr. Smith\n2,Miracle herb cures all diseases overnight,Unknown\n"
        items, meta = parse_batch_file(csv_data, "news.csv")
        self.assertEqual(len(items), 2)
        self.assertEqual(meta["detected_text_column"], "headline")
        self.assertEqual(items[0]["text"], "Space telescope discovers new galaxy")
        self.assertEqual(items[0]["metadata"]["author"], "Dr. Smith")
        self.assertEqual(items[1]["id"], "2")

    def test_detect_csv_text_column_longest_heuristic(self):
        headers = ["id", "tag", "custom_field"]
        sample_rows = [
            {"id": "1", "tag": "tech", "custom_field": "This is a detailed breaking news report about cyber security."},
            {"id": "2", "tag": "pol", "custom_field": "Officials announced new economic policy measures today."},
        ]
        best = detect_csv_text_column(headers, sample_rows)
        self.assertEqual(best, "custom_field")

    def test_parse_json_list_of_strings(self):
        json_data = b'["Headline number one about science", "Headline number two about sports"]'
        items, meta = parse_batch_file(json_data, "feed.json")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["text"], "Headline number one about science")
        self.assertEqual(items[1]["text"], "Headline number two about sports")

    def test_parse_json_list_of_objects(self):
        json_data = b'[{"id": "c1", "claim": "Moon is made of green cheese"}, {"id": "c2", "claim": "Earth revolves around the Sun"}]'
        items, meta = parse_batch_file(json_data, "claims.json")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["text"], "Moon is made of green cheese")
        self.assertEqual(items[0]["id"], "c1")

    def test_parse_json_wrapped_articles_dict(self):
        """Verifies multi-line formatted JSON dictionary wrapping an articles array parses all items."""
        json_data = b"""{
            "articles": [
                {"id": "c1", "text": "Space telescope discovers new galaxy", "author": "NASA"},
                {"id": "c2", "text": "Miracle cure debunked by medical review", "author": "WHO"}
            ]
        }"""
        items, meta = parse_batch_file(json_data, "news_feed.json")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "c1")
        self.assertEqual(items[0]["text"], "Space telescope discovers new galaxy")
        self.assertEqual(items[1]["id"], "c2")
        self.assertEqual(items[1]["text"], "Miracle cure debunked by medical review")

    def test_parse_jsonl(self):
        jsonl_data = b'{"id": "101", "text": "First article body text"}\n{"id": "102", "text": "Second article body text"}\n'
        items, meta = parse_batch_file(jsonl_data, "dump.jsonl")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "101")
        self.assertEqual(items[1]["text"], "Second article body text")

    def test_parse_jsonl_in_json_file_extension(self):
        """Verifies line-delimited records named with .json extension fallback cleanly to JSONL."""
        jsonl_data = b'{"id": "201", "text": "First line record"}\n{"id": "202", "text": "Second line record"}\n'
        items, meta = parse_batch_file(jsonl_data, "records.json")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["id"], "201")
        self.assertEqual(items[1]["id"], "202")

    def test_empty_json_raises_error(self):
        """Verifies empty or whitespace-only JSON payload raises BatchProcessingError."""
        with self.assertRaises(BatchProcessingError):
            parse_batch_file(b"   ", "empty.json")

    def test_parse_txt_paragraphs(self):
        txt_data = b"First long paragraph of news text that is definitely longer than ten chars.\n\nSecond paragraph of news text that is also quite long.\n"
        items, meta = parse_batch_file(txt_data, "notes.txt")
        self.assertEqual(len(items), 2)
        self.assertTrue("First long paragraph" in items[0]["text"])

    def test_unsupported_file_extension(self):
        with self.assertRaises(BatchProcessingError):
            parse_batch_file(b"data", "file.xyz")

    def test_empty_csv_raises_error(self):
        with self.assertRaises(BatchProcessingError):
            parse_batch_file(b"id,text\n", "empty.csv")

    @patch("News_Detection.batch_processor.analyze_text")
    def test_batch_job_execution_and_summary(self, mock_analyze):
        mock_analyze.side_effect = [
            {
                "is_fake": False,
                "soft_score": 92.5,
                "language_code": "en",
                "language_name": "English",
                "agreement": "3/3",
                "vote1": "AUTHENTIC",
                "score1": "92%",
                "vote2": "AUTHENTIC",
                "score2": "90%",
                "vote3": "AUTHENTIC",
                "score3": "95%",
                "vote4": "N/A",
                "score4": "N/A",
            },
            {
                "is_fake": True,
                "soft_score": 88.0,
                "language_code": "en",
                "language_name": "English",
                "agreement": "3/3",
                "vote1": "FAKE",
                "score1": "88%",
                "vote2": "FAKE",
                "score2": "85%",
                "vote3": "FAKE",
                "score3": "91%",
                "vote4": "N/A",
                "score4": "N/A",
            },
        ]

        manager = BatchJobManager()
        items = [
            {"id": "doc1", "text": "Genuine economic report", "metadata": {"source": "Bloomberg"}},
            {"id": "doc2", "text": "Miracle cure conspiracy", "metadata": {"source": "Blog"}},
        ]
        meta = {"filename": "test.csv", "total_extracted": 2}
        job_id = manager.create_job("test.csv", items, meta)

        manager.run_job_sync(job_id)

        job = manager.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["processed_items"], 2)
        self.assertEqual(job["summary"]["real_count"], 1)
        self.assertEqual(job["summary"]["fake_count"], 1)
        self.assertEqual(job["summary"]["real_pct"], 50.0)
        self.assertEqual(job["summary"]["fake_pct"], 50.0)

        # Verify analyze_text was called with enable_grounding=False by default (optimizing performance)
        for call in mock_analyze.call_args_list:
            self.assertEqual(call[1].get("enable_grounding"), False)

        # Test CSV export
        csv_str = manager.export_csv(job_id)
        self.assertIn("Bloomberg", csv_str)
        self.assertIn("REAL", csv_str)
        self.assertIn("FAKE", csv_str)

    @patch("News_Detection.batch_processor.analyze_text")
    def test_batch_job_explicit_grounding_flag(self, mock_analyze):
        mock_analyze.return_value = {
            "is_fake": False,
            "soft_score": 90.0,
            "language_code": "en",
            "language_name": "English",
            "agreement": "3/3",
            "vote1": "AUTHENTIC",
            "score1": "90%",
            "vote2": "AUTHENTIC",
            "score2": "90%",
            "vote3": "AUTHENTIC",
            "score3": "90%",
            "vote4": "N/A",
            "score4": "N/A",
        }
        manager = BatchJobManager()
        items = [{"id": "doc1", "text": "Claim text", "metadata": {}}]
        meta = {"filename": "test.txt", "total_extracted": 1}
        job_id = manager.create_job("test.txt", items, meta, enable_grounding=True)
        manager.run_job_sync(job_id)

        self.assertEqual(mock_analyze.call_args[1].get("enable_grounding"), True)

    @patch("News_Detection.batch_processor.analyze_text")
    def test_batch_job_unconfirmed_and_sha256_export(self, mock_analyze):
        mock_analyze.return_value = {
            "verdict": "UNCONFIRMED",
            "is_fake": False,
            "is_polarized": True,
            "polarization_index": 0.04,
            "claim_sha256": "abcdef1234567890" * 4,
            "soft_score": 52.0,
            "language_code": "en",
            "language_name": "English",
            "agreement": "2/3",
            "vote1": "FAKE",
            "score1": "52%",
            "vote2": "AUTHENTIC",
            "score2": "51%",
            "vote3": "AUTHENTIC",
            "score3": "50%",
            "vote4": "N/A",
            "score4": "N/A",
        }
        manager = BatchJobManager()
        items = [{"id": "doc1", "text": "Ambiguous claim with split consensus", "metadata": {"tag": "audit"}}]
        meta = {"filename": "audit.csv", "total_extracted": 1}
        job_id = manager.create_job("audit.csv", items, meta)
        manager.run_job_sync(job_id)

        job = manager.get_job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["summary"]["unconfirmed_count"], 1)
        self.assertEqual(job["summary"]["unconfirmed_pct"], 100.0)

        csv_str = manager.export_csv(job_id)
        self.assertIn("UNCONFIRMED", csv_str)
        self.assertIn("abcdef1234567890", csv_str)
        self.assertIn("claim_sha256", csv_str)
        self.assertIn("is_polarized", csv_str)


if __name__ == "__main__":
    unittest.main()
