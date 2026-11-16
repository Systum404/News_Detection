import unittest
from unittest.mock import MagicMock
from News_Detection.ensemble import (
    analyze_text,
    analyze_batch,
    extract_model_probabilities,
)


class TestEnsembleDetection(unittest.TestCase):

    def test_extract_model_probabilities_roberta(self):
        # RoBERTa Fake
        is_fake, p_fake, p_auth = extract_model_probabilities({"label": "FAKE", "score": 0.95}, "roberta")
        self.assertTrue(is_fake)
        self.assertAlmostEqual(p_fake, 0.95, places=3)
        self.assertAlmostEqual(p_auth, 0.05, places=3)

        # RoBERTa Real
        is_fake, p_fake, p_auth = extract_model_probabilities({"label": "REAL", "score": 0.90}, "roberta")
        self.assertFalse(is_fake)
        self.assertAlmostEqual(p_fake, 0.10, places=3)
        self.assertAlmostEqual(p_auth, 0.90, places=3)

    def test_extract_model_probabilities_tinybert(self):
        # TinyBERT: LABEL_1 is FAKE
        is_fake, p_fake, p_auth = extract_model_probabilities({"label": "LABEL_1", "score": 0.88}, "tinybert")
        self.assertTrue(is_fake)
        self.assertAlmostEqual(p_fake, 0.88, places=3)
        self.assertAlmostEqual(p_auth, 0.12, places=3)

        # TinyBERT: LABEL_0 is AUTHENTIC
        is_fake, p_fake, p_auth = extract_model_probabilities({"label": "LABEL_0", "score": 0.92}, "tinybert")
        self.assertFalse(is_fake)
        self.assertAlmostEqual(p_fake, 0.08, places=3)
        self.assertAlmostEqual(p_auth, 0.92, places=3)

    def test_extract_model_probabilities_bert_detect(self):
        # BERT-Detect: LABEL_0 is FAKE
        is_fake, p_fake, p_auth = extract_model_probabilities({"label": "LABEL_0", "score": 0.85}, "bert_detect")
        self.assertTrue(is_fake)
        self.assertAlmostEqual(p_fake, 0.85, places=3)
        self.assertAlmostEqual(p_auth, 0.15, places=3)

        # BERT-Detect: LABEL_1 is AUTHENTIC
        is_fake, p_fake, p_auth = extract_model_probabilities({"label": "LABEL_1", "score": 0.93}, "bert_detect")
        self.assertFalse(is_fake)
        self.assertAlmostEqual(p_fake, 0.07, places=3)
        self.assertAlmostEqual(p_auth, 0.93, places=3)

    def test_analyze_text_short_text_runs_all_models(self):
        """Verifies short claims (<15 words) actually run all 3 models and do not fake identical outputs."""
        mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.60}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.90}])

        res = analyze_text("Mango is a fruit.", pipes=(mock_p1, mock_p2, mock_p3))

        self.assertTrue(res["is_short"])
        # All 3 mock models were genuinely invoked
        self.assertTrue(mock_p1.called)
        self.assertTrue(mock_p2.called)
        self.assertTrue(mock_p3.called)

        # Check that individual votes are NOT duplicated/faked
        self.assertEqual(res["vote1"], "AUTHENTIC")
        self.assertEqual(res["vote2"], "FAKE")
        self.assertEqual(res["vote3"], "AUTHENTIC")

    def test_analyze_text_soft_consensus_overcomes_weak_hard_votes(self):
        """
        Demonstrates soft consensus:
        - Model 1 (RoBERTa, weight 0.40): AUTHENTIC with 99% confidence (P_auth = 0.99, P_fake = 0.01)
        - Model 2 (TinyBERT, weight 0.25): FAKE with 51% confidence (P_auth = 0.49, P_fake = 0.51)
        - Model 3 (BERT-Detect, weight 0.35): FAKE with 51% confidence (P_auth = 0.49, P_fake = 0.51)

        Hard voting would incorrectly declare FAKE (2 vs 1).
        Soft weighted consensus correctly recognizes AUTHENTIC because RoBERTa's high confidence outweighs weak guesses.
        """
        mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.99}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.51}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.51}])

        text = "A complete article text containing well over fifteen words to test the full ensemble pipeline under long-form news conditions."
        res = analyze_text(text, pipes=(mock_p1, mock_p2, mock_p3))

        # P_fake = (0.40 * 0.01) + (0.25 * 0.51) + (0.35 * 0.51) = 0.004 + 0.1275 + 0.1785 = 0.31
        # P_auth = 1.0 - 0.31 = 0.69 (69%)
        self.assertFalse(res["is_fake"])
        self.assertEqual(res["verdict"], "AUTHENTIC")
        self.assertIn("Agree", res["confidence"])
        self.assertGreater(res["soft_prob_auth"], 0.65)

    def test_analyze_text_empty_input(self):
        with self.assertRaises(ValueError):
            analyze_text("   ")

    def test_analyze_batch_empty_list(self):
        self.assertEqual(analyze_batch([]), [])

    def test_analyze_batch_vectorized_execution(self):
        """Verifies analyze_batch returns structured outputs for multiple articles."""
        mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}, {"label": "FAKE", "score": 0.90}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.92}, {"label": "LABEL_1", "score": 0.88}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.91}, {"label": "LABEL_0", "score": 0.89}])

        texts = [
            "Congress passes federal budget after bipartisan agreement in the Capitol.",
            "Scientists report discovering miracle cure in underground cavern.",
        ]
        results = analyze_batch(texts, pipes=(mock_p1, mock_p2, mock_p3), enable_grounding=False)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["verdict"], "AUTHENTIC")
        self.assertEqual(results[1]["verdict"], "FAKE")
        self.assertIn("category", results[0])
        self.assertIn("soft_score", results[0])

    def test_analyze_batch_handles_empty_elements(self):
        """Verifies analyze_batch handles empty elements without crashing valid elements."""
        mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])

        texts = ["Valid claim headline text here.", "", "   "]
        results = analyze_batch(texts, pipes=(mock_p1, mock_p2, mock_p3), enable_grounding=False)

        self.assertEqual(len(results), 3)
        self.assertIn(results[0]["verdict"], ["AUTHENTIC", "FAKE"])
        self.assertEqual(results[1]["verdict"], "ERROR")
        self.assertEqual(results[2]["verdict"], "ERROR")

    def test_ensemble_equivocal_neutral_zone(self):
        """Verifies borderline soft consensus (45%-55%) yields UNCONFIRMED verdict."""
        mock_p1 = MagicMock(return_value=[{"label": "FAKE", "score": 0.52}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.50}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.52}])

        text = "Ambiguous headline with conflicting facts reported across local media channels."
        res = analyze_text(text, pipes=(mock_p1, mock_p2, mock_p3))

        self.assertEqual(res["verdict"], "UNCONFIRMED")
        self.assertTrue(res["is_polarized"])
        self.assertEqual(res["raw_color"], "#fbbf24")
        self.assertIn("claim_sha256", res)
        self.assertEqual(len(res["claim_sha256"]), 64)

    def test_analyze_batch_includes_sha256_and_polarization(self):
        """Verifies analyze_batch outputs include deterministic claim_sha256 and is_polarized flag."""
        mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])

        texts = ["Space agency launches orbital mission successfully from southern launchpad."]
        results = analyze_batch(texts, pipes=(mock_p1, mock_p2, mock_p3), enable_grounding=False)

        self.assertEqual(len(results), 1)
        self.assertIn("claim_sha256", results[0])
        self.assertEqual(len(results[0]["claim_sha256"]), 64)
        self.assertIn("is_polarized", results[0])
        self.assertFalse(results[0]["is_polarized"])


if __name__ == "__main__":
    unittest.main()
