"""
Test Suite for Probability Calibration & Temperature Scaling
============================================================
Validates:
  - Inverse-sigmoid and logit numerical boundaries.
  - Temperature scaling properties: boundary invariance, monotonicity, softening.
  - Model-specific temperature calibrations (RoBERTa, TinyBERT, BERT-Detect, XLM-RoBERTa).
  - Statistical metrics: Brier score, Negative Log-Likelihood (NLL), Expected Calibration Error (ECE).
  - Golden-section temperature optimization.
  - FastAPI calibration diagnostic and evaluation endpoints.
"""

import math
import unittest
from unittest.mock import MagicMock

from News_Detection.calibration import (
    logit,
    sigmoid,
    scale_probability,
    scale_probability_distribution,
    TemperatureCalibrator,
    compute_brier_score,
    compute_nll,
    compute_ece,
    optimize_temperature,
)
from News_Detection.ensemble import analyze_text, extract_model_probabilities


class TestCalibrationMath(unittest.TestCase):
    """Validates core numerical routines and temperature transformation."""

    def test_logit_sigmoid_roundtrip(self):
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            z = logit(p)
            recovered = sigmoid(z)
            self.assertAlmostEqual(p, recovered, places=5)

    def test_extreme_boundary_protection(self):
        # Must not raise OverflowError or ValueError on extreme inputs
        z_high = logit(1.0)
        self.assertTrue(math.isfinite(z_high))
        p_high = sigmoid(1000.0)
        self.assertAlmostEqual(p_high, 1.0, places=5)

        z_low = logit(0.0)
        self.assertTrue(math.isfinite(z_low))
        p_low = sigmoid(-1000.0)
        self.assertAlmostEqual(p_low, 0.0, places=5)

    def test_decision_boundary_invariance(self):
        """p = 0.5 must remain strictly 0.5 across all temperatures."""
        for t in [0.5, 1.0, 1.45, 2.0, 5.0]:
            self.assertAlmostEqual(scale_probability(0.5, temperature=t), 0.5, places=5)

    def test_temperature_softening_behavior(self):
        """T > 1 must temper overconfident probabilities toward 0.5."""
        p_overconfident = 0.99
        p_calibrated = scale_probability(p_overconfident, temperature=1.5)

        self.assertLess(p_calibrated, p_overconfident)
        self.assertGreater(p_calibrated, 0.5)

        # For low probability (e.g. 0.01), T > 1 raises it toward 0.5
        p_low = 0.01
        p_calibrated_low = scale_probability(p_low, temperature=1.5)
        self.assertGreater(p_calibrated_low, p_low)
        self.assertLess(p_calibrated_low, 0.5)

    def test_monotonicity_preserved(self):
        """Higher initial probabilities must remain higher after scaling."""
        t = 1.65
        p1 = 0.70
        p2 = 0.85
        p3 = 0.95
        self.assertLess(scale_probability(p1, t), scale_probability(p2, t))
        self.assertLess(scale_probability(p2, t), scale_probability(p3, t))

    def test_invalid_temperature_raises_error(self):
        with self.assertRaises(ValueError):
            scale_probability(0.8, temperature=0.0)
        with self.assertRaises(ValueError):
            scale_probability(0.8, temperature=-1.5)


class TestTemperatureCalibrator(unittest.TestCase):
    """Validates the model-specific calibrator manager."""

    def setUp(self):
        self.calibrator = TemperatureCalibrator({
            "roberta": 1.45,
            "tinybert": 1.65,
            "bert_detect": 1.40,
        })

    def test_distribution_sums_to_one(self):
        p_fake, p_auth = scale_probability_distribution(0.92, 0.08, temperature=1.45)
        self.assertAlmostEqual(p_fake + p_auth, 1.0, places=5)

    def test_calibrate_probabilities_audit_metadata(self):
        p_fake, p_auth, info = self.calibrator.calibrate_probabilities(
            0.95, 0.05, model_type="tinybert"
        )
        self.assertEqual(info["model"], "tinybert")
        self.assertEqual(info["temperature"], 1.65)
        self.assertLess(p_fake, 0.95)
        self.assertAlmostEqual(p_fake + p_auth, 1.0, places=4)
        self.assertIn("adjustment_pct", info)


class TestCalibrationMetrics(unittest.TestCase):
    """Validates statistical metrics: Brier Score, NLL, and Expected Calibration Error (ECE)."""

    def test_brier_score_boundaries(self):
        # Perfect predictions: probability matches actual label exactly
        perfect_probs = [1.0, 0.0, 1.0, 0.0]
        labels = [1, 0, 1, 0]
        self.assertEqual(compute_brier_score(perfect_probs, labels), 0.0)

        # Worst predictions
        worst_probs = [0.0, 1.0, 0.0, 1.0]
        self.assertEqual(compute_brier_score(worst_probs, labels), 1.0)

    def test_nll_calculation(self):
        probs = [0.9, 0.1]
        labels = [1, 0]
        # -ln(0.9) ≈ 0.1054
        nll = compute_nll(probs, labels)
        self.assertAlmostEqual(nll, -math.log(0.9), places=3)

    def test_ece_computation(self):
        # Synthesize well-calibrated vs overconfident samples
        confidences = [0.95, 0.92, 0.88, 0.82, 0.75, 0.65, 0.60, 0.55]
        predictions = [1, 1, 1, 1, 0, 0, 1, 0]
        labels = [1, 1, 1, 0, 0, 0, 0, 0]

        ece_res = compute_ece(confidences, predictions, labels, num_bins=5)
        self.assertIn("ece", ece_res)
        self.assertIn("mce", ece_res)
        self.assertGreater(ece_res["ece"], 0.0)
        self.assertLessEqual(ece_res["ece"], 1.0)
        self.assertEqual(len(ece_res["bins"]), 5)

    def test_temperature_optimization(self):
        # Simulated overconfident model on validation set
        overconfident_probs = [0.99, 0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
        actual_labels = [1, 1, 1, 1, 0, 1, 0, 0]  # Only 5 of 8 are actually positive (~62.5%)

        opt = optimize_temperature(overconfident_probs, actual_labels, min_t=1.0, max_t=4.0)

        # Optimal temperature should be > 1.0 to soften the overconfidence
        self.assertGreater(opt["optimal_temperature"], 1.0)
        # Calibrated NLL should be strictly lower than initial NLL
        self.assertLess(opt["calibrated_nll"], opt["initial_nll"])
        self.assertGreater(opt["nll_reduction"], 0.0)


class TestEnsembleCalibrationIntegration(unittest.TestCase):
    """Validates that ensemble analysis incorporates probability calibration."""

    def test_extract_model_probabilities_with_calibration(self):
        # When calibrate=False (default): returns raw 0.96
        _, raw_fake, _ = extract_model_probabilities({"label": "FAKE", "score": 0.96}, "roberta", calibrate=False)
        self.assertEqual(raw_fake, 0.96)

        # When calibrate=True: returns calibrated (softened) probability
        _, cal_fake, _ = extract_model_probabilities({"label": "FAKE", "score": 0.96}, "roberta", calibrate=True)
        self.assertLess(cal_fake, 0.96)
        self.assertGreater(cal_fake, 0.85)

    def test_analyze_text_includes_calibration_metadata(self):
        mock_p1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}])
        mock_p2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        mock_p3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])

        result = analyze_text(
            "Economic indicators suggest steady recovery across global manufacturing markets.",
            pipes=(mock_p1, mock_p2, mock_p3),
            enable_calibration=True,
        )

        self.assertIn("calibration", result)
        self.assertTrue(result["calibration"]["enabled"])
        self.assertTrue(result["calibration"]["temperature_scaled"])
        self.assertIn("roberta", result["calibration"]["temperatures"])


class TestApiCalibrationEndpoints(unittest.TestCase):
    """Validates FastAPI calibration diagnostic and evaluation routes."""

    def test_calibration_routes(self):
        from fastapi.testclient import TestClient
        from News_Detection.health_api import app

        client = TestClient(app)

        # 1. GET /api/calibration/metrics
        metrics_resp = client.get("/api/calibration/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        data = metrics_resp.json()
        self.assertEqual(data["status"], "enabled")
        self.assertIn("roberta", data["temperatures"])

        # 2. POST /api/calibration/evaluate
        payload = {
            "samples": [
                {"confidence": 0.95, "prediction": 1, "actual": 1},
                {"confidence": 0.90, "prediction": 1, "actual": 1},
                {"confidence": 0.85, "prediction": 1, "actual": 0},
                {"confidence": 0.70, "prediction": 0, "actual": 0},
            ],
            "num_bins": 5,
        }
        eval_resp = client.post("/api/calibration/evaluate", json=payload)
        self.assertEqual(eval_resp.status_code, 200)
        eval_data = eval_resp.json()
        self.assertIn("evaluation", eval_data)
        self.assertIn("brier_score", eval_data)
        self.assertIn("optimization", eval_data)


if __name__ == "__main__":
    unittest.main()

