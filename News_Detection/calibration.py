"""
Probability Calibration & Temperature Scaling Engine
=====================================================
Post-processing statistical calibration for Transformer neural networks.
Mitigates overconfidence on out-of-distribution news claims by applying
temperature scaling:
    z_calibrated = logit(p) / T
    p_calibrated = sigmoid(z_calibrated)

Provides:
  - Inverse-sigmoid logit mapping with numerical stabilization.
  - Empirical model-specific temperature hyperparameters (RoBERTa, TinyBERT, BERT-Detect, XLM-RoBERTa).
  - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) evaluation.
  - Brier score computation and reliability diagram telemetry.
  - Optimal temperature tuning via Negative Log-Likelihood (NLL) scalar minimization.
"""

import os
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("news_detection.calibration")

# Model-specific temperature hyperparameters (T > 1 softens overconfident predictions)
DEFAULT_TEMPERATURES: Dict[str, float] = {
    "roberta": float(os.getenv("TEMPERATURE_ROBERTA", "1.45")),
    "tinybert": float(os.getenv("TEMPERATURE_TINYBERT", "1.65")),
    "bert_detect": float(os.getenv("TEMPERATURE_BERT_DETECT", "1.40")),
    "xlm_roberta": float(os.getenv("TEMPERATURE_XLM_ROBERTA", "1.50")),
    "ensemble": float(os.getenv("TEMPERATURE_ENSEMBLE", "1.20")),
}


def logit(p: float, eps: float = 1e-7) -> float:
    """
    Computes inverse sigmoid (logit) function with numerical boundary clamping.
    p must be in (0, 1).
    """
    p_clamped = max(eps, min(1.0 - eps, float(p)))
    return math.log(p_clamped / (1.0 - p_clamped))


def sigmoid(z: float) -> float:
    """Computes standard logistic sigmoid with numerical overflow protection."""
    if z >= 0:
        exp_neg = math.exp(-z)
        return 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)


def scale_probability(prob: float, temperature: float = 1.0) -> float:
    """
    Applies temperature scaling to a single probability value.
    Preserves monotonicity and binary decision boundaries (p=0.5 -> 0.5).
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive (> 0), got {temperature}")
    if abs(temperature - 1.0) < 1e-6:
        return max(0.0, min(1.0, float(prob)))

    z = logit(prob)
    z_scaled = z / temperature
    return sigmoid(z_scaled)


def scale_probability_distribution(
    p_fake: float,
    p_auth: float,
    temperature: float = 1.0,
) -> Tuple[float, float]:
    """
    Calibrates a binary probability distribution (p_fake, p_auth) using temperature scaling.
    Ensures p_fake_calibrated + p_auth_calibrated == 1.0.
    """
    p_fake_scaled = scale_probability(p_fake, temperature)
    return p_fake_scaled, 1.0 - p_fake_scaled


class TemperatureCalibrator:
    """
    Manages temperature calibration for ensemble components and aggregate consensus.
    """

    def __init__(self, temperatures: Optional[Dict[str, float]] = None):
        self.temperatures = dict(DEFAULT_TEMPERATURES)
        if temperatures:
            self.temperatures.update(temperatures)

    def get_temperature(self, model_type: str) -> float:
        """Returns the configured temperature for a model type, defaulting to 1.0."""
        return self.temperatures.get(model_type.lower(), 1.0)

    def set_temperature(self, model_type: str, temperature: float):
        """Updates the temperature parameter for a specific model."""
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperatures[model_type.lower()] = float(temperature)

    def calibrate_probabilities(
        self,
        p_fake: float,
        p_auth: float,
        model_type: str,
        override_temperature: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calibrates model output probabilities and returns audit metadata.
        """
        temp = override_temperature if override_temperature is not None else self.get_temperature(model_type)
        calibrated_fake, calibrated_auth = scale_probability_distribution(p_fake, p_auth, temp)

        info = {
            "model": model_type,
            "temperature": temp,
            "raw_prob_fake": round(p_fake, 4),
            "raw_prob_auth": round(p_auth, 4),
            "calibrated_prob_fake": round(calibrated_fake, 4),
            "calibrated_prob_auth": round(calibrated_auth, 4),
            "adjustment_pct": round((calibrated_fake - p_fake) * 100, 2),
        }
        return calibrated_fake, calibrated_auth, info


# --- CALIBRATION EVALUATION & STATISTICAL METRICS ---

def compute_brier_score(probabilities: List[float], labels: List[int]) -> float:
    """
    Computes mean squared error between predicted probabilities and binary ground truth.
    Lower is better (0.0 is perfect calibration and resolution).
    """
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must be non-empty and of equal length.")

    total_error = sum((p - y) ** 2 for p, y in zip(probabilities, labels))
    return round(total_error / len(probabilities), 4)


def compute_nll(probabilities: List[float], labels: List[int], eps: float = 1e-7) -> float:
    """
    Computes average Negative Log-Likelihood (cross-entropy).
    """
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must be non-empty and of equal length.")

    nll = 0.0
    for p, y in zip(probabilities, labels):
        p_clamped = max(eps, min(1.0 - eps, float(p)))
        if y == 1:
            nll -= math.log(p_clamped)
        else:
            nll -= math.log(1.0 - p_clamped)
    return round(nll / len(probabilities), 4)


def compute_ece(
    confidences: List[float],
    predictions: List[int],
    labels: List[int],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    Partitions samples into M equal-width confidence intervals [0, 1/M], [1/M, 2/M]...
    and evaluates the weighted absolute difference between confidence and accuracy.
    """
    if not (len(confidences) == len(predictions) == len(labels)):
        raise ValueError("All input arrays must have the same length.")

    n = len(confidences)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    bin_width = 1.0 / num_bins
    bins_data = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width

        # Collect items in bin [bin_lower, bin_upper] (inclusive on right for last bin)
        indices = [
            idx
            for idx, conf in enumerate(confidences)
            if (bin_lower <= conf < bin_upper) or (i == num_bins - 1 and conf == bin_upper)
        ]

        bin_count = len(indices)
        if bin_count > 0:
            avg_conf = sum(confidences[idx] for idx in indices) / bin_count
            correct = sum(1 for idx in indices if predictions[idx] == labels[idx])
            accuracy = correct / bin_count
            diff = abs(accuracy - avg_conf)

            ece += (bin_count / n) * diff
            mce = max(mce, diff)

            bins_data.append({
                "bin": i + 1,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": bin_count,
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(accuracy, 4),
                "calibration_gap": round(diff, 4),
            })
        else:
            bins_data.append({
                "bin": i + 1,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": 0,
                "avg_confidence": 0.0,
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "total_samples": n,
        "num_bins": num_bins,
        "bins": bins_data,
    }


def optimize_temperature(
    probabilities: List[float],
    labels: List[int],
    min_t: float = 0.5,
    max_t: float = 5.0,
    steps: int = 100,
) -> Dict[str, Any]:
    """
    Finds the optimal temperature parameter T* that minimizes Negative Log-Likelihood (NLL)
    using golden-section / bounded grid search.
    """
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must match in length.")

    initial_nll = compute_nll(probabilities, labels)
    initial_brier = compute_brier_score(probabilities, labels)

    best_temp = 1.0
    best_nll = initial_nll

    # Dense scalar sweep over candidate temperatures
    step_size = (max_t - min_t) / steps
    for step in range(steps + 1):
        temp = min_t + (step * step_size)
        scaled_probs = [scale_probability(p, temp) for p in probabilities]
        nll = compute_nll(scaled_probs, labels)
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    best_probs = [scale_probability(p, best_temp) for p in probabilities]
    calibrated_brier = compute_brier_score(best_probs, labels)

    return {
        "optimal_temperature": round(best_temp, 3),
        "initial_nll": initial_nll,
        "calibrated_nll": best_nll,
        "nll_reduction": round(initial_nll - best_nll, 4),
        "initial_brier_score": initial_brier,
        "calibrated_brier_score": calibrated_brier,
    }


# Global Default Calibrator Instance
calibrator = TemperatureCalibrator()

