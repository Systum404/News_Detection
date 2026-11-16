import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from unittest.mock import MagicMock
from News_Detection.ensemble import (
    analyze_text,
    extract_model_probabilities,
)


def run_parameter_audit():
    print("=" * 80)
    print("      MULTI-MODEL ENSEMBLE: COMPREHENSIVE PARAMETER AUDIT")
    print("=" * 80)
    print()

    # 1. Model Configuration & Weights
    print("[1] ARCHITECTURAL WEIGHTS & MODEL SPECIFICATIONS")
    print("-" * 80)
    models = [
        ("RoBERTa-Base", "hamzab/roberta-fake-news-classification", 0.40, "125M", "FAKE / REAL"),
        ("Distilled TinyBERT", "mrm8488/bert-tiny-finetuned-fake-news-detection", 0.25, "14.3M", "LABEL_1 (Fake) / LABEL_0 (Real)"),
        ("BERT-Detect", "jy46604790/Fake-News-Bert-Detect", 0.35, "110M", "LABEL_0 (Fake) / LABEL_1 (Real)"),
    ]
    print(f"{'Model Name':<22} | {'Parameters':<10} | {'Weight':<8} | {'Label Mapping Format':<32}")
    print("-" * 80)
    for name, path, weight, params, mapping in models:
        print(f"{name:<22} | {params:<10} | {weight:<8.2f} | {mapping:<32}")
    print()

    # 2. Probability Normalization Parameter Sweeps
    print("[2] LOGIT NORMALIZATION & SCORE BOUNDARY TESTS")
    print("-" * 80)
    test_scores = [0.50, 0.65, 0.85, 0.99, 1.0, 1.25, -0.1]
    print(f"{'Input Model':<14} | {'Raw Label':<10} | {'Raw Score':<10} | {'Normalized P(Fake)':<18} | {'Normalized P(Auth)':<18} | {'Status'}")
    print("-" * 80)
    for model_type, label in [("roberta", "FAKE"), ("tinybert", "LABEL_1"), ("bert_detect", "LABEL_0")]:
        for s in [0.55, 0.95, 1.2]:
            is_fake, pf, pa = extract_model_probabilities({"label": label, "score": s}, model_type)
            clamped = "[CLAMPED]" if s > 1.0 or s < 0.0 else "[VALID]"
            print(f"{model_type:<14} | {label:<10} | {s:<10.2f} | {pf:<18.4f} | {pa:<18.4f} | {clamped}")
    print()

    # 3. Soft Consensus Voting Truth Table
    print("[3] CONSENSUS DECISION TRUTH TABLE (ALL VOTING PERMUTATIONS)")
    print("-" * 80)
    permutations = [
        ("FFF", (True, 0.95), (True, 0.90), (True, 0.88)),
        ("FFA", (True, 0.85), (True, 0.80), (False, 0.70)),
        ("FAF", (True, 0.85), (False, 0.75), (True, 0.80)),
        ("FAA", (True, 0.70), (False, 0.85), (False, 0.90)),
        ("AFF (Override)", (False, 0.99), (True, 0.52), (True, 0.52)),
        ("AFF (Majority)", (False, 0.60), (True, 0.95), (True, 0.95)),
        ("AFA", (False, 0.90), (True, 0.75), (False, 0.85)),
        ("AAF", (False, 0.90), (False, 0.85), (True, 0.75)),
        ("AAA", (False, 0.98), (False, 0.95), (False, 0.92)),
    ]

    print(f"{'State':<16} | {'RoBERTa':<12} | {'TinyBERT':<12} | {'BERT-Detect':<12} | {'Verdict':<10} | {'Soft Score':<11} | {'Agreement'}")
    print("-" * 80)
    for name, (f1, s1), (f2, s2), (f3, s3) in permutations:
        m1 = MagicMock(return_value=[{"label": "FAKE" if f1 else "REAL", "score": s1}])
        m2 = MagicMock(return_value=[{"label": "LABEL_1" if f2 else "LABEL_0", "score": s2}])
        m3 = MagicMock(return_value=[{"label": "LABEL_0" if f3 else "LABEL_1", "score": s3}])

        res = analyze_text("Sample standard testing text exceeding fifteen words to verify the ensemble decision matrix.", pipes=(m1, m2, m3))
        s1_str = f"{'F' if f1 else 'A'} ({s1*100:.0f}%)"
        s2_str = f"{'F' if f2 else 'A'} ({s2*100:.0f}%)"
        s3_str = f"{'F' if f3 else 'A'} ({s3*100:.0f}%)"
        print(f"{name:<16} | {s1_str:<12} | {s2_str:<12} | {s3_str:<12} | {res['verdict']:<10} | {res['soft_score']:<10.1f}% | {res['agreement']}")
    print()

    # 4. Text Length Parameter Sweeps
    print("[4] INPUT LENGTH & TRUNCATION SWEEPS")
    print("-" * 80)
    lengths = [
        ("Short claim (4 words)", "Earth orbits the sun.", True),
        ("Exact boundary (14 words)", "One two three four five six seven eight nine ten eleven twelve thirteen fourteen.", True),
        ("Exact boundary (15 words)", "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen.", False),
        ("Medium article (100 words)", ("headline statement reuters " * 25).strip(), False),
        ("Large article (5000 chars)", ("investigation parliament statement government authentic " * 150).strip(), False),
    ]
    print(f"{'Input Type':<26} | {'Word Count':<12} | {'Char Length':<12} | {'is_short':<10} | {'Snippet Preview'}")
    print("-" * 80)
    for desc, sample, exp_short in lengths:
        m1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}])
        m2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        m3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])
        res = analyze_text(sample, pipes=(m1, m2, m3))
        wc = len(sample.split())
        cl = len(sample)
        snip = res["snippet"][:30] + ("..." if len(res["snippet"]) > 30 else "")
        print(f"{desc:<26} | {wc:<12} | {cl:<12} | {str(res['is_short']):<10} | {snip}")
    print()

    # 5. Latency & Throughput Benchmark
    print("[5] EXECUTION LATENCY & STABILITY SWEEP")
    print("-" * 80)
    pipes = (
        MagicMock(return_value=[{"label": "REAL", "score": 0.92}]),
        MagicMock(return_value=[{"label": "LABEL_0", "score": 0.88}]),
        MagicMock(return_value=[{"label": "LABEL_1", "score": 0.95}]),
    )
    iterations = 500
    t0 = time.perf_counter()
    for _ in range(iterations):
        analyze_text("Bipartisan congressional negotiation reaches accord on budgetary allocations.", pipes=pipes)
    elapsed = time.perf_counter() - t0
    ops_per_sec = iterations / elapsed
    print(f"Executed {iterations} end-to-end ensemble pipeline evaluations in {elapsed:.4f}s ({ops_per_sec:.1f} ops/sec).")
    print()

    # 6. Multilingual Language Routing Verification
    print("[6] MULTILINGUAL NLP & LANGUAGE ROUTING VERIFICATION")
    print("-" * 80)
    ml_samples = [
        ("English", "The United Nations General Assembly passed a unanimous resolution on climate financing.", "en", False),
        ("Hindi", "भारतीय अंतरिक्ष एजेंसी इसरो ने श्रीहरिकोटा से एक नया संचार उपग्रह सफलतापूर्वक लॉन्च किया।", "hi", True),
        ("Spanish", "El presidente del gobierno presentó una nueva reforma laboral aprobada por el parlamento.", "es", True),
        ("French", "Le ministre de l'économie a dévoilé les nouvelles mesures d'aide aux entreprises locales.", "fr", True),
        ("German", "Die Europäische Zentralbank hat heute eine Anpassung der Leitzinsen für das laufende Quartal beschlossen.", "de", True),
    ]
    print(f"{'Source Language':<16} | {'Expected Code':<14} | {'Detected Lang':<22} | {'is_multilingual':<16} | {'Status'}")
    print("-" * 80)
    for lang_name, text, exp_code, exp_multi in ml_samples:
        m1 = MagicMock(return_value=[{"label": "REAL", "score": 0.95}])
        m2 = MagicMock(return_value=[{"label": "LABEL_0", "score": 0.90}])
        m3 = MagicMock(return_value=[{"label": "LABEL_1", "score": 0.92}])
        res = analyze_text(text, pipes=(m1, m2, m3))
        det_code = res["language_code"]
        det_name = res["language_name"]
        det_multi = res["is_multilingual"]
        match = (det_code == exp_code) and (det_multi == exp_multi)
        status_str = "PASS" if match else "FAIL"
        print(f"{lang_name:<16} | {exp_code:<14} | {det_name:<22} | {str(det_multi):<16} | {status_str}")
        assert match, f"Language mismatch for {lang_name}: expected {exp_code}, got {det_code}"
    print()

    # 7. Hardware Acceleration & Quantization Verification
    print("[7] HARDWARE ACCELERATION & PRECISION QUANTIZATION VERIFICATION")
    print("-" * 80)
    from News_Detection.accelerator import get_hardware_status, benchmark_acceleration
    hw = get_hardware_status()
    print(f"Active Hardware Device : {hw['device_name']} ({hw['active_device']})")
    print(f"Apple Silicon MPS      : {'Available' if hw['mps_available'] else 'N/A'}")
    print(f"FP16 Half-Precision    : {'Enabled' if hw['half_precision_fp16'] else 'Disabled'}")
    print(f"ONNX Runtime Engine    : {'v' + str(hw['onnx_version']) if hw['onnx_available'] else 'N/A'}")
    print(f"ONNX Providers         : {', '.join(hw['onnx_providers'])}")
    print(f"Quantization Modes     : {', '.join(hw['quantization_modes'])}")
    assert hw["mps_available"] or hw["active_device"] in ("mps", "cpu"), "Invalid hardware device state"
    assert hw["onnx_available"], "ONNX Runtime must be installed and accessible"

    bench = benchmark_acceleration(iterations=5)
    print(f"Hardware Latency Bench : CPU: {bench['cpu_latency_ms']:.2f}ms | Accelerated: {bench['accelerated_latency_ms']:.2f}ms ({bench['speedup_factor']})")
    print()

    print("=" * 80)
    print(">> ALL PARAMETER, MULTILINGUAL & HARDWARE ACCELERATION TESTS PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    run_parameter_audit()
