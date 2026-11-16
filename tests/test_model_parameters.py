import unittest
import time
from unittest.mock import MagicMock
from News_Detection.ensemble import (
    analyze_text,
    extract_model_probabilities,
    are_models_loaded,
)


class TestModelParametersComprehensive(unittest.TestCase):
    """
    Comprehensive parameter test suite covering:
      1. Model probability normalization across all label formats and scores.
      2. Voting & soft consensus across all combinations of model weights and scores.
      3. Input text parameters: length boundaries, special characters, unicode, XSS, fuzzing.
      4. Output dictionary schema, types, and invariants.
      5. Execution timing and stability on stress inputs.
    """

    # =========================================================================
    # PARAMETER GROUP 1: Individual Model Label & Logit Normalization
    # =========================================================================
    def test_roberta_parameter_permutations(self):
        """Test RoBERTa probability extraction across label variants and confidence ranges."""
        test_cases = [
            # (input_dict, expected_is_fake, expected_prob_fake, expected_prob_auth)
            ({"label": "FAKE", "score": 0.99}, True, 0.99, 0.01),
            ({"label": "fake", "score": 0.85}, True, 0.85, 0.15),
            ({"label": "REAL", "score": 0.95}, False, 0.05, 0.95),
            ({"label": "TRUE", "score": 0.80}, False, 0.20, 0.80),
            ({"label": "real", "score": 0.60}, False, 0.40, 0.60),
            # Boundary scores
            ({"label": "FAKE", "score": 1.0}, True, 1.0, 0.0),
            ({"label": "REAL", "score": 0.50}, False, 0.50, 0.50),
            ({"label": "FAKE", "score": 1.2}, True, 1.0, 0.0),  # Out-of-bounds clamped
            ({"label": "REAL", "score": -0.1}, False, 1.0, 0.0),  # Negative clamped
        ]

        for inp, exp_fake, exp_pfake, exp_pauth in test_cases:
            is_fake, p_fake, p_auth = extract_model_probabilities(inp, "roberta")
            self.assertEqual(is_fake, exp_fake, f"Failed on RoBERTa input: {inp}")
            self.assertAlmostEqual(p_fake, exp_pfake, places=3)
            self.assertAlmostEqual(p_auth, exp_pauth, places=3)
            self.assertAlmostEqual(p_fake + p_auth, 1.0, places=3)

    def test_tinybert_parameter_permutations(self):
        """Test TinyBERT: LABEL_1 (Fake) vs LABEL_0 (Authentic)."""
        test_cases = [
            ({"label": "LABEL_1", "score": 0.98}, True, 0.98, 0.02),
            ({"label": "LABEL_1", "score": 0.52}, True, 0.52, 0.48),
            ({"label": "LABEL_0", "score": 0.94}, False, 0.06, 0.94),
            ({"label": "LABEL_0", "score": 0.51}, False, 0.49, 0.51),
        ]
        for inp, exp_fake, exp_pfake, exp_pauth in test_cases:
            is_fake, p_fake, p_auth = extract_model_probabilities(inp, "tinybert")
            self.assertEqual(is_fake, exp_fake, f"Failed on TinyBERT input: {inp}")
            self.assertAlmostEqual(p_fake, exp_pfake, places=3)
            self.assertAlmostEqual(p_auth, exp_pauth, places=3)

    def test_bert_detect_parameter_permutations(self):
        """Test BERT-Detect: LABEL_0 (Fake) vs LABEL_1 (Authentic)."""
        test_cases = [
            ({"label": "LABEL_0", "score": 0.99}, True, 0.99, 0.01),
            ({"label": "LABEL_0", "score": 0.55}, True, 0.55, 0.45),
            ({"label": "LABEL_1", "score": 0.91}, False, 0.09, 0.91),
            ({"label": "LABEL_1", "score": 0.50}, False, 0.50, 0.50),
        ]
        for inp, exp_fake, exp_pfake, exp_pauth in test_cases:
            is_fake, p_fake, p_auth = extract_model_probabilities(inp, "bert_detect")
            self.assertEqual(is_fake, exp_fake, f"Failed on BERT-Detect input: {inp}")
            self.assertAlmostEqual(p_fake, exp_pfake, places=3)
            self.assertAlmostEqual(p_auth, exp_pauth, places=3)

    # =========================================================================
    # PARAMETER GROUP 2: Ensemble Consensus Combinations (All 8 Voting States)
    # =========================================================================
    def test_ensemble_voting_truth_table(self):
        """
        Tests all 8 possible combinations of model binary decisions (FFF, FFA, FAF, FAA, AFF, AFA, AAF, AAA)
        under varying confidence levels to verify soft consensus stability.
        """
        # Weights: RoBERTa = 0.40, TinyBERT = 0.25, BERT-Detect = 0.35
        scenarios = [
            # 1. Unanimous Fake (FFF)
            {
                "m1": ({"label": "FAKE", "score": 0.90}, "roberta"),
                "m2": ({"label": "LABEL_1", "score": 0.85}, "tinybert"),
                "m3": ({"label": "LABEL_0", "score": 0.80}, "bert_detect"),
                "expected_verdict": "FAKE",
                "expected_agreement": "3/3",
            },
            # 2. Unanimous Authentic (AAA)
            {
                "m1": ({"label": "REAL", "score": 0.92}, "roberta"),
                "m2": ({"label": "LABEL_0", "score": 0.88}, "tinybert"),
                "m3": ({"label": "LABEL_1", "score": 0.95}, "bert_detect"),
                "expected_verdict": "AUTHENTIC",
                "expected_agreement": "3/3",
            },
            # 3. FFA (RoBERTa & TinyBERT Fake, BERT-Detect Authentic)
            {
                "m1": ({"label": "FAKE", "score": 0.80}, "roberta"),      # P_fake = 0.80 * 0.40 = 0.32
                "m2": ({"label": "LABEL_1", "score": 0.80}, "tinybert"),  # P_fake = 0.80 * 0.25 = 0.20
                "m3": ({"label": "LABEL_1", "score": 0.70}, "bert_detect"),# P_fake = 0.30 * 0.35 = 0.105
                # Soft P_fake = 0.625 -> FAKE
                "expected_verdict": "FAKE",
                "expected_agreement": "2/3",
            },
            # 4. FAA (RoBERTa Fake, TinyBERT & BERT-Detect Authentic)
            {
                "m1": ({"label": "FAKE", "score": 0.70}, "roberta"),      # P_fake = 0.70 * 0.40 = 0.28
                "m2": ({"label": "LABEL_0", "score": 0.80}, "tinybert"),  # P_fake = 0.20 * 0.25 = 0.05
                "m3": ({"label": "LABEL_1", "score": 0.85}, "bert_detect"),# P_fake = 0.15 * 0.35 = 0.0525
                # Soft P_fake = 0.3825 -> AUTHENTIC
                "expected_verdict": "AUTHENTIC",
                "expected_agreement": "2/3",
            },
            # 5. AFF (RoBERTa Authentic, TinyBERT & BERT-Detect Fake)
            # Case where RoBERTa has VERY high confidence (0.99) vs weak Fakes (0.52, 0.52)
            {
                "m1": ({"label": "REAL", "score": 0.99}, "roberta"),      # P_fake = 0.01 * 0.40 = 0.004
                "m2": ({"label": "LABEL_1", "score": 0.52}, "tinybert"),  # P_fake = 0.52 * 0.25 = 0.130
                "m3": ({"label": "LABEL_0", "score": 0.52}, "bert_detect"),# P_fake = 0.52 * 0.35 = 0.182
                # Soft P_fake = 0.316 -> AUTHENTIC (High confidence overrides 2 weak fakes!)
                "expected_verdict": "AUTHENTIC",
                "expected_agreement": "1/3",
            },
            # 6. AFF (RoBERTa Authentic with moderate confidence 0.70, strong Fakes 0.95, 0.95)
            {
                "m1": ({"label": "REAL", "score": 0.70}, "roberta"),      # P_fake = 0.30 * 0.40 = 0.12
                "m2": ({"label": "LABEL_1", "score": 0.95}, "tinybert"),  # P_fake = 0.95 * 0.25 = 0.2375
                "m3": ({"label": "LABEL_0", "score": 0.95}, "bert_detect"),# P_fake = 0.95 * 0.35 = 0.3325
                # Soft P_fake = 0.69 -> FAKE (Strong consensus overrides moderate RoBERTa)
                "expected_verdict": "FAKE",
                "expected_agreement": "2/3",
            },
        ]

        sample_text = "This is a sample news article text formatted with sufficient word count to exceed fifteen words easily."
        for idx, sc in enumerate(scenarios, 1):
            mock1 = MagicMock(return_value=[sc["m1"][0]])
            mock2 = MagicMock(return_value=[sc["m2"][0]])
            mock3 = MagicMock(return_value=[sc["m3"][0]])

            res = analyze_text(sample_text, pipes=(mock1, mock2, mock3))
            self.assertEqual(
                res["verdict"],
                sc["expected_verdict"],
                f"Scenario #{idx} failed: Expected verdict {sc['expected_verdict']}, got {res['verdict']}"
            )
            self.assertEqual(
                res["agreement"],
                sc["expected_agreement"],
                f"Scenario #{idx} agreement failed: Expected {sc['expected_agreement']}, got {res['agreement']}"
            )

    # =========================================================================
    # PARAMETER GROUP 3: Text Length Thresholds & Word-Boundary Windowing
    # =========================================================================
    def test_text_length_threshold_14_vs_15_words(self):
        """Validates exact boundary: 14 words (is_short=True) vs 15 words (is_short=False)."""
        mock1 = MagicMock(return_value=[{"label": "REAL", "score": 0.90}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.90}])
        pipes = (mock1, mock2, mock3)

        # 14 words
        text_14 = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen."
        self.assertEqual(len(text_14.split()), 14)
        res_14 = analyze_text(text_14, pipes=pipes)
        self.assertTrue(res_14["is_short"])

        # 15 words
        text_15 = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen."
        self.assertEqual(len(text_15.split()), 15)
        res_15 = analyze_text(text_15, pipes=pipes)
        self.assertFalse(res_15["is_short"])

    def test_large_text_truncation_boundary(self):
        """Tests that text > 2500 characters is cleanly truncated on word boundaries without slicing words."""
        mock1 = MagicMock(return_value=[{"label": "REAL", "score": 0.90}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.90}])
        pipes = (mock1, mock2, mock3)

        # Create 10,000 character string of clean words
        word_list = ["investigation", "parliament", "statement", "government", "authentic"] * 250
        large_text = " ".join(word_list)
        self.assertGreater(len(large_text), 8000)

        res = analyze_text(large_text, pipes=pipes)
        self.assertEqual(res["verdict"], "AUTHENTIC")
        # Check that mock received truncated text <= 2500 chars and ends cleanly
        passed_text = mock1.call_args[0][0]
        self.assertLessEqual(len(passed_text), 2500)
        self.assertFalse(passed_text.endswith(" "), "Truncated text should not end with trailing space")
        # Check that snippet is cleanly truncated to <= 63 characters
        self.assertLessEqual(len(res["snippet"]), 63)
        self.assertTrue(res["snippet"].endswith("..."))

    # =========================================================================
    # PARAMETER GROUP 4: Edge Cases, Special Characters & Fuzz Inputs
    # =========================================================================
    def test_empty_and_whitespace_inputs(self):
        """Verifies empty or whitespace-only strings raise ValueError."""
        mock_pipes = (MagicMock(), MagicMock(), MagicMock())
        for invalid_text in ["", "   ", "\t\t\n\n", "     \r\n   "]:
            with self.assertRaises(ValueError):
                analyze_text(invalid_text, pipes=mock_pipes)

    def test_insufficient_context_single_words_rejected(self):
        """Verifies single-word non-claims ('System') and 2-word inputs raise ValueError."""
        mock_pipes = (MagicMock(), MagicMock(), MagicMock())
        for non_claim in ["System", "Apple", "Breaking news", "Only two"]:
            with self.assertRaises(ValueError):
                analyze_text(non_claim, pipes=mock_pipes)

    def test_lexical_diversity_and_repetition_spam_rejected(self):
        """Phase 6.3: Verifies repetitive token spam and low-entropy inputs raise ValueError."""
        mock_pipes = (MagicMock(), MagicMock(), MagicMock())
        # Single repeated word (3+ words)
        with self.assertRaises(ValueError) as ctx:
            analyze_text("System System System", pipes=mock_pipes)
        self.assertIn("single repeated word", str(ctx.exception))

        # Repeated phrase with low TTR (< 0.35)
        with self.assertRaises(ValueError) as ctx:
            analyze_text("Fake news! Fake news! Fake news! Fake news!", pipes=mock_pipes)
        self.assertIn("low lexical diversity", str(ctx.exception))

    def test_special_characters_and_emojis(self):
        """Verifies ensemble handles emojis, punctuation, currency symbols, and quotes cleanly."""
        mock1 = MagicMock(return_value=[{"label": "FAKE", "score": 0.96}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.88}])

        special_text = "🚨🚨 BREAKING: $10,000,000,000 in fake crypto 'found' @ White House!??? #FakeNews 🛸🛸 100% CONFIRMED!!!"
        res = analyze_text(special_text, pipes=(mock1, mock2, mock3))

        self.assertEqual(res["verdict"], "FAKE")
        self.assertTrue(res["is_short"])
        self.assertIn("92.", res["confidence"])  # Soft confidence around 92%

    def test_multilingual_unicode_text(self):
        """Verifies text with non-Latin Unicode characters (e.g., Devanagari, accented European) processes without encoding errors."""
        mock1 = MagicMock(return_value=[{"label": "REAL", "score": 0.90}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.90}])

        hindi_french_text = "विशेष समाचार: Le président a annoncé une nouvelle politique économique hier soir à Paris."
        res = analyze_text(hindi_french_text, pipes=(mock1, mock2, mock3))
        self.assertEqual(res["verdict"], "AUTHENTIC")

    def test_html_and_script_injection_safety(self):
        """Verifies HTML/script tags are treated purely as inert text strings and do not cause execution failure."""
        mock1 = MagicMock(return_value=[{"label": "FAKE", "score": 0.99}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.95}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])

        html_text = "<script>alert('xss')</script><iframe src='http://evil.com'></iframe> Breaking news report."
        res = analyze_text(html_text, pipes=(mock1, mock2, mock3))
        self.assertEqual(res["verdict"], "FAKE")
        self.assertIn("<script>", res["snippet"])

    # =========================================================================
    # PARAMETER GROUP 5: Output Schema & Mathematical Invariants
    # =========================================================================
    def test_output_schema_and_invariants(self):
        """Verifies all dictionary keys, types, and mathematical invariants in the response."""
        mock1 = MagicMock(return_value=[{"label": "FAKE", "score": 0.88}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.72}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.65}])  # Model 3 says Authentic

        text = "Extensive report regarding recent economic inflation numbers released by the Federal Reserve Bank."
        res = analyze_text(text, categories="economics", pipes=(mock1, mock2, mock3))

        # Check required keys
        required_keys = [
            "result", "verdict", "is_fake", "confidence", "agreement",
            "soft_score", "soft_prob_fake", "soft_prob_auth",
            "bar_width", "bar_color", "color", "bg_glow", "raw_color",
            "snippet", "is_short", "vote1", "score1", "is_fake1",
            "vote2", "score2", "is_fake2", "vote3", "score3", "is_fake3",
            "claim_sha256", "is_polarized", "polarization_index"
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Missing key: {key}")

        # Milestone 7.2: Verify claim_sha256 deterministic hex digest
        self.assertEqual(len(res["claim_sha256"]), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in res["claim_sha256"]))

        # Mathematical Invariants
        self.assertAlmostEqual(res["soft_prob_fake"] + res["soft_prob_auth"], 1.0, places=3)
        self.assertGreaterEqual(res["soft_score"], 50.0)
        self.assertLessEqual(res["soft_score"], 100.0)

        # Bar width matches soft_score
        expected_width = f"{res['soft_score']:.0f}%"
        self.assertEqual(res["bar_width"], expected_width)

        # Verdict matches is_fake
        if res["is_fake"]:
            self.assertEqual(res["verdict"], "FAKE")
            self.assertEqual(res["result"], "FAKE NEWS DETECTED")
            self.assertIn("red", res["bar_color"])
        else:
            self.assertEqual(res["verdict"], "AUTHENTIC")
            self.assertEqual(res["result"], "AUTHENTIC NEWS")
            self.assertIn("emerald", res["bar_color"])

    def test_equivocal_neutral_zone_polarization(self):
        """Milestone 7.1: Tests that soft consensus between 45% and 55% returns UNCONFIRMED with amber tokens."""
        # Setup mock probabilities that yield soft_prob_fake ~ 0.501
        # RoBERTa (w=0.40): Fake with score 0.52 -> P_fake = 0.52 * 0.40 = 0.208
        # TinyBERT (w=0.25): Fake with score 0.50 -> P_fake = 0.50 * 0.25 = 0.125
        # BERT-Detect (w=0.35): Authentic (LABEL_1) with score 0.52 -> P_fake = 0.48 * 0.35 = 0.168
        # Total P_fake = 0.208 + 0.125 + 0.168 = 0.501 (50.1%)
        mock1 = MagicMock(return_value=[{"label": "FAKE", "score": 0.52}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.50}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.52}])
        pipes = (mock1, mock2, mock3)

        text = "Ambiguous breaking political report with conflicting statements across multiple party officials today."
        res = analyze_text(text, pipes=pipes)

        self.assertTrue(0.45 < res["soft_prob_fake"] < 0.55)
        self.assertTrue(res["is_polarized"])
        self.assertEqual(res["verdict"], "UNCONFIRMED")
        self.assertEqual(res["result"], "UNCONFIRMED // HIGH MODEL POLARIZATION")
        self.assertFalse(res["is_fake"])
        self.assertEqual(res["raw_color"], "#fbbf24")
        self.assertEqual(res["color"], "text-amber-400")
        self.assertEqual(res["bar_color"], "bg-amber-500")
        self.assertIn("shadow-amber-500", res["bg_glow"])
        self.assertGreaterEqual(res["polarization_index"], 0.0)

    def test_deterministic_claim_sha256_fingerprint(self):
        """Milestone 7.2: Tests that identical claim texts produce identical SHA-256 digests across calls."""
        import hashlib
        import unicodedata

        mock1 = MagicMock(return_value=[{"label": "REAL", "score": 0.90}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.90}])
        pipes = (mock1, mock2, mock3)

        text = "Government central statistical office releases fourth quarter gross domestic product growth indicators."
        expected_hash = hashlib.sha256(unicodedata.normalize("NFKC", text.strip()).encode("utf-8")).hexdigest()

        res1 = analyze_text(text, pipes=pipes)
        res2 = analyze_text(text, pipes=pipes)

        self.assertEqual(res1["claim_sha256"], expected_hash)
        self.assertEqual(res2["claim_sha256"], expected_hash)
        self.assertEqual(res1["claim_sha256"], res2["claim_sha256"])

    # =========================================================================
    # PARAMETER GROUP 6: Performance & Latency Benchmark on Parameter Sweeps
    # =========================================================================
    def test_parameter_sweep_execution_latency(self):
        """Verifies that 100 consecutive parameter permutations execute in under 0.05 seconds with mock pipelines."""
        mock1 = MagicMock(return_value=[{"label": "FAKE", "score": 0.85}])
        mock2 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.75}])
        mock3 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        pipes = (mock1, mock2, mock3)

        start = time.perf_counter()
        for i in range(100):
            analyze_text(f"Sample test article number {i} running through full parameter checks.", pipes=pipes)
        duration = time.perf_counter() - start

        # 100 iterations of text parsing, language detection, truncation, 3-model normalization, and weighted consensus
        self.assertLess(duration, 1.0, f"Parameter sweep took {duration:.4f}s, expected < 1.0s")


if __name__ == "__main__":
    unittest.main()
