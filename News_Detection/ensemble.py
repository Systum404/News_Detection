import logging
import threading
import concurrent.futures
import re
import hashlib
import unicodedata
from typing import Optional, Tuple, Dict, Any, List

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

from News_Detection.accelerator import get_optimal_device
from News_Detection.grounding import ground_claims

logger = logging.getLogger("news_detection.ensemble")

# --- AI ENSEMBLE CONFIGURATION (THREAD-SAFE & LAZY-LOADED) ---
logger.info("Initializing ensemble news detection engine")

pipe1 = None  # RoBERTa Base
pipe2 = None  # Distilled TinyBERT
pipe3 = None  # BERT-Detect
pipe4 = None  # XLM-RoBERTa Multilingual (zodumair/xlmr-fake-news-multilingual)
_model_lock = threading.Lock()
_active_device_name = "Optimized CPU (ARM64 / x86)"
_active_precision = "FP32"

# Reusable concurrent inference executor for parallel forward passes across models
_inference_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="ensemble_worker",
)

# Reusable concurrent grounding executor for non-blocking decoupled web I/O
_grounding_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=6,
    thread_name_prefix="grounding_worker",
)

# --- MULTILINGUAL NLP CONFIGURATION ---
LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "mr": "Marathi (मराठी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "pt": "Portuguese (Português)",
    "it": "Italian (Italiano)",
    "tr": "Turkish (Türkçe)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "gu": "Gujarati (ગુજરાતી)",
    "ur": "Urdu (اردو)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "vi": "Vietnamese (Tiếng Việt)",
    "id": "Indonesian (Bahasa Indonesia)",
    "sv": "Swedish (Svenska)",
}


def detect_language(text: str) -> Tuple[str, str]:
    """
    Detects ISO-639 language code and formatted display name using statistical n-gram profiles.
    Returns (code, display_name). Defaults gracefully to ('en', 'English') upon error or ambiguity.
    """
    clean = text.strip() if text else ""
    if not clean or len(clean) < 4:
        return "en", "English"
    try:
        from langdetect import detect
        code = str(detect(clean)).lower()
        name = LANGUAGE_MAP.get(code, f"International ({code.upper()})")
        return code, name
    except Exception:
        return "en", "English"


def get_pipes() -> Tuple[object, object, object]:
    """Thread-safe lazy-loading of the 3 Transformer pipelines with hardware acceleration."""
    global pipe1, pipe2, pipe3, _active_device_name, _active_precision
    if pipe1 is None or pipe2 is None or pipe3 is None:
        with _model_lock:
            if pipeline is None:
                raise ImportError(
                    "The 'transformers' library is required to load neural network models. "
                    "Please install it using: pip install transformers torch"
                )
            
            device, dtype, device_name = get_optimal_device()
            _active_device_name = device_name
            _active_precision = "FP16" if "FP16" in device_name else "FP32"
            
            pipe_kwargs = {}
            if str(device) != "cpu" and hasattr(device, "type") and device.type in ("mps", "cuda"):
                pipe_kwargs["device"] = device
                if dtype is not None:
                    pipe_kwargs["dtype"] = dtype

            logger.info(f"Hardware accelerator configured: {device_name} (Precision: {_active_precision})")
            
            if pipe1 is None:
                logger.info("Loading RoBERTa Base model (hamzab/roberta-fake-news-classification)...")
                pipe1 = pipeline("text-classification", model="hamzab/roberta-fake-news-classification", **pipe_kwargs)
            if pipe2 is None:
                logger.info("Loading Distilled TinyBERT model (mrm8488/bert-tiny-finetuned-fake-news-detection)...")
                pipe2 = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection", **pipe_kwargs)
            if pipe3 is None:
                logger.info("Loading BERT-Detect model (jy46604790/Fake-News-Bert-Detect)...")
                pipe3 = pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect", **pipe_kwargs)
            logger.info("Ensemble neural pipelines initialized and ready")
    return pipe1, pipe2, pipe3


def get_xlmr_pipe() -> Optional[object]:
    """Thread-safe lazy-loading of the specialized XLM-RoBERTa multilingual transformer."""
    global pipe4
    if pipe4 is None:
        with _model_lock:
            if pipe4 is None:
                if pipeline is None:
                    return None
                try:
                    device, dtype, device_name = get_optimal_device()
                    pipe_kwargs = {}
                    if str(device) != "cpu" and hasattr(device, "type") and device.type in ("mps", "cuda"):
                        pipe_kwargs["device"] = device
                        if dtype is not None:
                            pipe_kwargs["dtype"] = dtype

                    logger.info(f"Loading XLM-RoBERTa Multilingual model on {device_name}...")
                    pipe4 = pipeline(
                        "text-classification",
                        model="zodumair/xlmr-fake-news-multilingual",
                        **pipe_kwargs,
                    )
                    logger.info("XLM-RoBERTa multilingual model ready")
                except Exception as e:
                    logger.warning(f"XLM-RoBERTa lazy-load deferred ({e})")
                    return None
    return pipe4


def are_models_loaded() -> bool:
    """Returns True if the primary neural network pipelines are loaded in memory."""
    return all(p is not None for p in (pipe1, pipe2, pipe3))


def _exec_pipe(pipe: object, text: str) -> Optional[Dict[str, Any]]:
    """Safe runner executing a model forward pass with keyword fallback for mocks."""
    if pipe is None:
        return None
    try:
        res = pipe(text, truncation=True, max_length=512)
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        return res
    except TypeError:
        # Fallback for synthetic/mock pipes in unit tests that do not take kwargs
        res = pipe(text)
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        return res


def warmup_ensemble() -> bool:
    """
    Pre-warms all neural transformer pipelines and hardware execution kernels (MPS/CUDA).
    Eliminates cold-start latency for the first user query.
    """
    try:
        p1, p2, p3 = get_pipes()
        sample_text = "Warmup benchmark claim to initialize neural kernels."
        _exec_pipe(p1, sample_text)
        _exec_pipe(p2, sample_text)
        _exec_pipe(p3, sample_text)

        # Warm up news categorizer
        from News_Detection.categorizer import predict_news_category
        predict_news_category(sample_text)
        logger.info("Neural ensemble pre-warmed & ready (0ms cold start)")
        return True
    except Exception as exc:
        logger.warning(f"Ensemble warmup deferred or skipped: {exc}")
        return False


def extract_model_probabilities(
    pipe_result: Dict[str, Any],
    model_type: str,
    calibrate: bool = False,
    override_temperature: Optional[float] = None,
) -> Tuple[bool, float, float]:
    """
    Normalizes heterogeneous model output logits into (is_fake, prob_fake, prob_authentic).
    Optionally applies post-processing temperature scaling to correct overconfident predictions.

    Model Label Specifications:
      - roberta: 'FAKE' / 'REAL'
      - tinybert: 'LABEL_1' (FAKE) / 'LABEL_0' (REAL)
      - bert_detect: 'LABEL_0' (FAKE) / 'LABEL_1' (REAL)
      - xlm_roberta: 'FAKE' / 'REAL'
    """
    raw_label = str(pipe_result.get("label", "")).upper()
    top_score = float(pipe_result.get("score", 0.0))
    top_score = max(0.0, min(1.0, top_score))

    if model_type in ("roberta", "xlm_roberta"):
        is_fake = "FAKE" in raw_label
    elif model_type == "tinybert":
        is_fake = raw_label == "LABEL_1"
    elif model_type == "bert_detect":
        is_fake = raw_label == "LABEL_0"
    else:
        is_fake = "FAKE" in raw_label

    if is_fake:
        prob_fake = top_score
        prob_authentic = 1.0 - top_score
    else:
        prob_authentic = top_score
        prob_fake = 1.0 - top_score

    if calibrate:
        try:
            from News_Detection.calibration import calibrator
            prob_fake, prob_authentic, _ = calibrator.calibrate_probabilities(
                prob_fake,
                prob_authentic,
                model_type=model_type,
                override_temperature=override_temperature,
            )
        except Exception:
            pass

    return is_fake, prob_fake, prob_authentic


def _batch_exec(pipe: object, text_list: List[str], bsize: int = 32) -> List[Any]:
    """
    Executes a vectorized batch forward pass on a Transformer pipeline.
    Falls back to item-by-item execution for synthetic mock fixtures in unit tests.
    """
    if pipe is None or not text_list:
        return []
    try:
        raw = pipe(text_list, batch_size=bsize, truncation=True, max_length=512)
        if isinstance(raw, list):
            res = []
            for item in raw:
                if isinstance(item, list) and len(item) > 0:
                    res.append(item[0])
                else:
                    res.append(item)
            return res
        return [raw] * len(text_list)
    except TypeError:
        # Fallback for synthetic/mock functions in unit tests that only accept single string arguments
        out = []
        for t in text_list:
            r = _exec_pipe(pipe, t)
            out.append(r)
        return out


def _aggregate_model_verdicts(
    sanitized: str,
    res1: Any,
    res2: Any,
    res3: Any,
    res4: Any,
    lang_code: str,
    lang_name: str,
    is_multilingual: bool,
    category_pred: Dict[str, Any],
    fact_check: Dict[str, Any],
    enable_calibration: bool,
) -> Dict[str, Any]:
    """Combines heterogeneous model outputs into continuous soft consensus probabilities."""
    words = sanitized.split()
    is_short = len(words) < 15

    # Model 1: RoBERTa Base
    is_fake1, p_fake1, p_auth1 = extract_model_probabilities(res1 or {}, "roberta", calibrate=enable_calibration)
    vote1 = "FAKE" if is_fake1 else "AUTHENTIC"
    score1_val = max(p_fake1, p_auth1)
    score1_str = f"{score1_val * 100:.1f}%"

    # Model 2: Distilled TinyBERT
    is_fake2, p_fake2, p_auth2 = extract_model_probabilities(res2 or {}, "tinybert", calibrate=enable_calibration)
    vote2 = "FAKE" if is_fake2 else "AUTHENTIC"
    score2_val = max(p_fake2, p_auth2)
    score2_str = f"{score2_val * 100:.1f}%"

    # Model 3: BERT-Detect
    is_fake3, p_fake3, p_auth3 = extract_model_probabilities(res3 or {}, "bert_detect", calibrate=enable_calibration)
    vote3 = "FAKE" if is_fake3 else "AUTHENTIC"
    score3_val = max(p_fake3, p_auth3)
    score3_str = f"{score3_val * 100:.1f}%"

    # Model 4: XLM-RoBERTa Multilingual
    has_xlmr = False
    vote4 = "N/A"
    score4_str = "N/A"
    is_fake4 = False
    p_fake4 = 0.5
    p_auth4 = 0.5

    if res4 is not None:
        try:
            is_fake4, p_fake4, p_auth4 = extract_model_probabilities(res4, "xlm_roberta", calibrate=enable_calibration)
            vote4 = "FAKE" if is_fake4 else "AUTHENTIC"
            score4_val = max(p_fake4, p_auth4)
            score4_str = f"{score4_val * 100:.1f}%"
            has_xlmr = True
        except Exception:
            has_xlmr = False

    # --- SOFT CONSENSUS ENSEMBLE (PROBABILITY-WEIGHTED VOTING) ---
    if has_xlmr:
        w1, w2, w3, w4 = 0.25, 0.15, 0.20, 0.40
        soft_prob_fake = (w1 * p_fake1) + (w2 * p_fake2) + (w3 * p_fake3) + (w4 * p_fake4)
        soft_prob_auth = (w1 * p_auth1) + (w2 * p_auth2) + (w3 * p_auth3) + (w4 * p_auth4)
        votes = [vote1, vote2, vote3, vote4]
        total_models = 4
    else:
        w1, w2, w3 = 0.40, 0.25, 0.35
        soft_prob_fake = (w1 * p_fake1) + (w2 * p_fake2) + (w3 * p_fake3)
        soft_prob_auth = (w1 * p_auth1) + (w2 * p_auth2) + (w3 * p_auth3)
        votes = [vote1, vote2, vote3]
        total_models = 3

    fake_votes = votes.count("FAKE")
    auth_votes = votes.count("AUTHENTIC")

    # Milestone 7.2: Deterministic SHA-256 Claim Fingerprint for chain-of-custody forensic auditability
    normalized_claim = unicodedata.normalize("NFKC", sanitized.strip())
    claim_sha256 = hashlib.sha256(normalized_claim.encode("utf-8")).hexdigest()

    # Milestone 7.1: Equivocal Neutral Zone (45.0% - 55.0%)
    # Prevents binary flips on razor-thin consensus (e.g. 50.4% fake vs 49.6% authentic)
    is_polarized = 0.45 < soft_prob_fake < 0.55
    polarization_index = round(abs(soft_prob_fake - 0.5) * 2, 4)

    if soft_prob_fake >= 0.55:
        is_fake = True
        winning_prob = soft_prob_fake
        agreement_count = fake_votes
        verdict = "FAKE"
        final_res = "FAKE NEWS DETECTED"
        final_color = "text-red-400"
        bg_glow = "shadow-red-500/20"
        bar_color = "bg-red-500"
        raw_color = "#f87171"
    elif soft_prob_fake <= 0.45:
        is_fake = False
        winning_prob = soft_prob_auth
        agreement_count = auth_votes
        verdict = "AUTHENTIC"
        final_res = "AUTHENTIC NEWS"
        final_color = "text-emerald-400"
        bg_glow = "shadow-emerald-500/20"
        bar_color = "bg-emerald-500"
        raw_color = "#34d399"
    else:
        # Equivocal Neutral Zone: High model polarization
        is_fake = False
        winning_prob = max(soft_prob_fake, soft_prob_auth)
        agreement_count = max(fake_votes, auth_votes)
        verdict = "UNCONFIRMED"
        final_res = "UNCONFIRMED // HIGH MODEL POLARIZATION"
        final_color = "text-amber-400"
        bg_glow = "shadow-amber-500/20"
        bar_color = "bg-amber-500"
        raw_color = "#fbbf24"

    conf_pct = winning_prob * 100.0
    bar_width = f"{conf_pct:.0f}%"
    final_conf = f"{conf_pct:.1f}% ({agreement_count}/{total_models} Agree)"
    snippet = sanitized[:60] + "..." if len(sanitized) > 60 else sanitized

    return {
        "result": final_res,
        "is_fake": is_fake,
        "verdict": verdict,
        "is_polarized": is_polarized,
        "polarization_index": polarization_index,
        "claim_sha256": claim_sha256,
        "confidence": final_conf,
        "agreement": f"{agreement_count}/{total_models}",
        "soft_score": round(conf_pct, 2),
        "soft_prob_fake": round(soft_prob_fake, 4),
        "soft_prob_auth": round(soft_prob_auth, 4),
        "bar_width": bar_width,
        "bar_color": bar_color,
        "color": final_color,
        "bg_glow": bg_glow,
        "raw_color": raw_color,
        "snippet": snippet,
        "is_short": is_short,
        "vote1": vote1,
        "score1": score1_str,
        "is_fake1": is_fake1,
        "vote2": vote2,
        "score2": score2_str,
        "is_fake2": is_fake2,
        "vote3": vote3,
        "score3": score3_str,
        "is_fake3": is_fake3,
        "vote4": vote4,
        "score4": score4_str,
        "is_fake4": is_fake4,
        "label1": vote1,
        "label2": vote2,
        "label3": vote3,
        "label4": vote4,
        "has_xlmr": has_xlmr,
        "total_models": total_models,
        "language_code": lang_code,
        "language_name": lang_name,
        "is_multilingual": is_multilingual,
        "routing_engine": (
            "3-Model Transformer Ensemble"
            if not is_multilingual
            else (
                f"Multilingual 4-Model Cross-Lingual Ensemble ({lang_name})"
                if has_xlmr
                else f"Multilingual Cross-Lingual Pipeline ({lang_name})"
            )
        ),
        "acceleration_device": _active_device_name,
        "hardware_accelerated": "MPS" in _active_device_name or "CUDA" in _active_device_name,
        "precision": _active_precision,
        "fact_check": fact_check,
        "category": category_pred,
        "calibration": {
            "enabled": enable_calibration,
            "temperature_scaled": bool(enable_calibration),
            "temperatures": {
                "roberta": 1.45,
                "tinybert": 1.65,
                "bert_detect": 1.40,
                "xlm_roberta": 1.50,
                "ensemble": 1.20,
            } if enable_calibration else {},
        },
        "cached": False,
        "cache_hit": False,
    }


def validate_claim_text(text: str) -> str:
    """
    Validates that input text is substantive, contains at least 3 words,
    and does not exhibit degenerate token repetition or low lexical entropy.
    Returns stripped and sanitized text.
    """
    sanitized = text.strip() if text else ""
    if not sanitized:
        raise ValueError("Article text cannot be empty.")

    words = [w for w in sanitized.split() if w.strip()]
    if len(words) < 3:
        raise ValueError(
            f"Input contains only {len(words)} word{'s' if len(words) != 1 else ''}. "
            "News authenticity verification requires a complete claim, headline, or article body (minimum 3 words)."
        )

    # Clean tokens for lexical diversity evaluation
    tokens = [re.sub(r"[^\w\s]", "", w.lower()) for w in words]
    tokens = [t for t in tokens if t]

    # 1. Single repeated word check (e.g. "System System System")
    if len(tokens) >= 3 and len(set(tokens)) < 2:
        raise ValueError(
            f"Input consists of a single repeated word ('{tokens[0]}'). "
            "News authenticity analysis requires a complete claim or headline with substantive context."
        )

    # 2. Type-Token Ratio (TTR) repetition spam guard
    if 8 <= len(tokens) <= 60:
        ttr = len(set(tokens)) / len(tokens)
        if ttr < 0.35:
            raise ValueError(
                f"Input exhibits low lexical diversity (TTR: {ttr:.2f} < 0.35) due to excessive word repetition. "
                "News authenticity analysis requires authentic, substantive journalism or complete claims."
            )
    elif len(tokens) > 60:
        from collections import Counter
        most_common_word, count = Counter(tokens).most_common(1)[0]
        if count / len(tokens) > 0.35:
            raise ValueError(
                f"Input exhibits excessive word repetition ('{most_common_word}' constitutes {count / len(tokens):.1%} of the text). "
                "News authenticity analysis requires authentic, substantive journalism."
            )

    return sanitized


def analyze_text(
    text: str,
    categories: Optional[str] = None,
    pipes: Optional[Tuple[object, ...]] = None,
    enable_grounding: Optional[bool] = None,
    use_cache: bool = True,
    enable_calibration: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Analyzes input text using a multi-model Transformer ensemble with Soft Probability Consensus
    and real-time live Fact-Checking & Web Grounding (RAG).
    
    Both short claims and full articles are genuinely evaluated across all neural models.
    Returns structured results including continuous confidence metrics, model breakdowns,
    and live web evidence citations.
    """
    sanitized = validate_claim_text(text)

    if enable_calibration is None:
        enable_calibration = (pipes is None)

    # 0. Prediction Cache Lookup (Fast Path)
    resolved_grounding = (pipes is None) if enable_grounding is None else bool(enable_grounding)
    if use_cache and pipes is None:
        try:
            from News_Detection.storage import prediction_cache
            cached_result = prediction_cache.get(
                sanitized,
                categories=categories,
                enable_grounding=resolved_grounding,
            )
            if cached_result is not None:
                return cached_result
        except Exception:
            pass

    # 1. Automatic Language Identification & Cross-Lingual Routing
    lang_code, lang_name = detect_language(sanitized)
    is_multilingual = lang_code != "en"

    # 2. Topic & News Category Prediction
    try:
        from News_Detection.categorizer import predict_news_category
        category_pred = predict_news_category(sanitized)
        if categories:
            category_pred["user_specified"] = categories
    except Exception:
        category_pred = {
            "category": "International News",
            "confidence": 50.0,
            "distribution": {},
            "matched_signals": [],
        }

    # 3. Decoupled Concurrent Web Fact-Checking & Grounding (Phase 4)
    if enable_grounding is None:
        enable_grounding = (pipes is None)

    grounding_fut = None
    if enable_grounding:
        # Offload web grounding network I/O to background thread concurrently with neural inference
        grounding_fut = _grounding_pool.submit(ground_claims, sanitized, lang_code)

    if len(sanitized) > 2500:
        truncated_text = sanitized[:2500].rsplit(" ", 1)[0]
    else:
        truncated_text = sanitized

    # Model Pipeline Retrieval
    p4 = None
    if pipes is not None:
        if len(pipes) >= 4:
            p1, p2, p3, p4 = pipes[:4]
        else:
            p1, p2, p3 = pipes[:3]
    else:
        p1, p2, p3 = get_pipes()
        if is_multilingual:
            p4 = get_xlmr_pipe()

    # --- ADAPTIVE CONCURRENT / VECTORIZED DISPATCH ---
    is_mps = "MPS" in _active_device_name

    if not is_mps:
        fut1 = _inference_pool.submit(_exec_pipe, p1, truncated_text)
        fut2 = _inference_pool.submit(_exec_pipe, p2, truncated_text)
        fut3 = _inference_pool.submit(_exec_pipe, p3, truncated_text)
        fut4 = (
            _inference_pool.submit(_exec_pipe, p4, truncated_text)
            if (is_multilingual and p4 is not None)
            else None
        )
        res1 = fut1.result()
        res2 = fut2.result()
        res3 = fut3.result()
        res4 = fut4.result() if fut4 is not None else None
    else:
        res1 = _exec_pipe(p1, truncated_text)
        res2 = _exec_pipe(p2, truncated_text)
        res3 = _exec_pipe(p3, truncated_text)
        res4 = _exec_pipe(p4, truncated_text) if (is_multilingual and p4 is not None) else None

    # Retrieve concurrent web grounding result with resilient timeout guard
    if grounding_fut is not None:
        try:
            fact_check = grounding_fut.result(timeout=4.0)
        except Exception as exc:
            logger.warning(f"Concurrent web grounding failed or timed out: {exc}")
            fact_check = {
                "corroboration_score": 0.0,
                "corroboration_status": "TIMEOUT",
                "summary": "External web grounding timed out; neural inference verdict retained.",
                "sources": [],
                "authoritative_count": 0,
                "total_found": 0,
                "search_query_used": "",
                "cached": False,
            }
    else:
        fact_check = {
            "corroboration_score": 0.0,
            "corroboration_status": "SKIPPED",
            "summary": "Live web grounding skipped in test/mock mode.",
            "sources": [],
            "authoritative_count": 0,
            "total_found": 0,
            "search_query_used": "",
            "cached": False,
        }

    result = _aggregate_model_verdicts(
        sanitized=sanitized,
        res1=res1,
        res2=res2,
        res3=res3,
        res4=res4,
        lang_code=lang_code,
        lang_name=lang_name,
        is_multilingual=is_multilingual,
        category_pred=category_pred,
        fact_check=fact_check,
        enable_calibration=enable_calibration,
    )

    if use_cache and pipes is None:
        try:
            from News_Detection.storage import prediction_cache
            cache_key = prediction_cache.set(
                sanitized,
                result,
                categories=categories,
                enable_grounding=resolved_grounding,
            )
            result["cache_key"] = cache_key
        except Exception:
            pass

    return result


def analyze_batch(
    texts: List[str],
    categories: Optional[str] = None,
    pipes: Optional[Tuple[object, ...]] = None,
    enable_grounding: bool = False,
    use_cache: bool = True,
    enable_calibration: Optional[bool] = None,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    High-throughput vectorized batch analysis for bulk document auditing.
    Processes multiple texts simultaneously via PyTorch tensor batching.
    """
    if not texts:
        return []

    if enable_calibration is None:
        enable_calibration = (pipes is None)

    results: List[Optional[Dict[str, Any]]] = [None] * len(texts)

    # 1. First Pass: Cache Lookup
    if use_cache and pipes is None:
        try:
            from News_Detection.storage import prediction_cache
            for idx, text in enumerate(texts):
                sanitized = text.strip() if text else ""
                if not sanitized:
                    continue
                cached = prediction_cache.get(
                    sanitized,
                    categories=categories,
                    enable_grounding=enable_grounding,
                )
                if cached is not None:
                    results[idx] = cached
        except Exception:
            pass

    uncached_indices = [i for i, r in enumerate(results) if r is None]
    if not uncached_indices:
        return [r for r in results if r is not None]

    # 2. Extract texts and metadata for uncached items
    valid_work: List[Tuple[int, str]] = []
    for orig_i in uncached_indices:
        t = texts[orig_i].strip() if texts[orig_i] else ""
        if not t:
            results[orig_i] = {
                "result": "ERROR: EMPTY TEXT",
                "is_fake": False,
                "verdict": "ERROR",
                "confidence": "0.0%",
                "agreement": "0/0",
                "soft_score": 0.0,
                "error": "Article text cannot be empty.",
            }
        else:
            valid_work.append((orig_i, t))

    if not valid_work:
        return [r for r in results if r is not None]

    valid_orig_indices = [item[0] for item in valid_work]
    valid_texts = [item[1] for item in valid_work]

    # Language Identification and Topic Categorization
    langs = [detect_language(t) for t in valid_texts]
    cats = []
    for t in valid_texts:
        try:
            from News_Detection.categorizer import predict_news_category
            cp = predict_news_category(t)
            if categories:
                cp["user_specified"] = categories
            cats.append(cp)
        except Exception:
            cats.append({"category": "International News", "confidence": 50.0})

    truncated_texts = [
        t[:2500].rsplit(" ", 1)[0] if len(t) > 2500 else t
        for t in valid_texts
    ]

    # 3. Model Pipeline Retrieval
    p4 = None
    if pipes is not None:
        p1, p2, p3 = pipes[:3]
        if len(pipes) >= 4:
            p4 = pipes[3]
    else:
        p1, p2, p3 = get_pipes()
        if any(l[0] != "en" for l in langs):
            p4 = get_xlmr_pipe()

    # 4. Vectorized Batch Forward Passes across all 3 primary models
    res1_list = _batch_exec(p1, truncated_texts, bsize=batch_size)
    res2_list = _batch_exec(p2, truncated_texts, bsize=batch_size)
    res3_list = _batch_exec(p3, truncated_texts, bsize=batch_size)

    # 5. Multilingual Model (XLM-RoBERTa) on non-English subset
    res4_map: Dict[int, Any] = {}
    if p4 is not None:
        multi_indices = [k for k, (lcode, _) in enumerate(langs) if lcode != "en"]
        if multi_indices:
            multi_texts = [truncated_texts[k] for k in multi_indices]
            res4_batch = _batch_exec(p4, multi_texts, bsize=batch_size)
            for k, r4 in zip(multi_indices, res4_batch):
                res4_map[k] = r4

    # 6. Construct Results & Update Multi-Tier Cache
    for k, (orig_i, sanitized) in enumerate(zip(valid_orig_indices, valid_texts)):
        lcode, lname = langs[k]
        ismulti = lcode != "en"
        cpred = cats[k]
        r1 = res1_list[k] if k < len(res1_list) else {}
        r2 = res2_list[k] if k < len(res2_list) else {}
        r3 = res3_list[k] if k < len(res3_list) else {}
        r4 = res4_map.get(k)

        fact_check = {
            "corroboration_score": 0.0,
            "corroboration_status": "SKIPPED",
            "summary": "Live web grounding skipped in batch inference mode.",
            "sources": [],
            "authoritative_count": 0,
            "total_found": 0,
            "search_query_used": "",
        }

        item_res = _aggregate_model_verdicts(
            sanitized=sanitized,
            res1=r1,
            res2=r2,
            res3=r3,
            res4=r4,
            lang_code=lcode,
            lang_name=lname,
            is_multilingual=ismulti,
            category_pred=cpred,
            fact_check=fact_check,
            enable_calibration=enable_calibration,
        )

        # Store in Multi-Tier Cache
        if use_cache and pipes is None:
            try:
                from News_Detection.storage import prediction_cache
                ckey = prediction_cache.set(
                    sanitized,
                    item_res,
                    categories=categories,
                    enable_grounding=enable_grounding,
                )
                item_res["cache_key"] = ckey
            except Exception:
                pass

        results[orig_i] = item_res

    return [r for r in results if r is not None]

