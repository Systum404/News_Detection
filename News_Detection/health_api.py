import os
import re
import asyncio
import logging
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Response,
    Request,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .ensemble import analyze_text, are_models_loaded
from .rate_limiter import SlidingWindowRateLimiter, RateLimitMiddleware
from .batch_processor import parse_batch_file, batch_job_manager, BatchProcessingError

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("news_detection.api")

# Concurrency & Safeguard Thresholds (configurable via environment variables)
INFERENCE_TIMEOUT_SECONDS = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "30.0"))
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(15 * 1024 * 1024)))  # 15MB
CORS_ORIGINS_ENV = os.getenv("CORS_ORIGINS", "*")

# Global Concurrency Semaphore to prevent CPU thrashing & OOM spikes during bursts
_inference_semaphore: Optional[asyncio.Semaphore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes server state and lifecycle resources with asynchronous model pre-warming."""
    global _inference_semaphore
    _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    logger.info(
        f"News Detection API Initialized | Timeout: {INFERENCE_TIMEOUT_SECONDS}s | "
        f"Max Concurrent Inferences: {MAX_CONCURRENT_INFERENCES} | "
        f"Rate Limit: {RATE_LIMIT_REQUESTS} req / {RATE_LIMIT_WINDOW_SECONDS}s"
    )
    # Background pre-warming of neural models to eliminate cold start (0ms latency for 1st query)
    try:
        from News_Detection.ensemble import warmup_ensemble
        asyncio.create_task(asyncio.to_thread(warmup_ensemble))
    except Exception as exc:
        logger.warning(f"Startup ensemble pre-warming deferred: {exc}")

    yield
    logger.info("News Detection API shutting down...")
    try:
        from News_Detection.grounding import close_http_clients
        close_http_clients()
    except Exception:
        pass


app = FastAPI(
    title="News Detection API",
    description=(
        "Production REST service for multi-model deep learning fake news classification. "
        "Engineered with event-loop non-blocking thread offloading, timeout safeguards, "
        "concurrency bounds, and IP-based rate limiting."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

# --- 1. CORS & SECURITY HEADERS MIDDLEWARE ---
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

if CORS_ORIGINS_ENV and CORS_ORIGINS_ENV != "*":
    origins = [o.strip() for o in CORS_ORIGINS_ENV.split(",") if o.strip()]
    allow_wildcard = "*" in origins
else:
    origins = DEFAULT_ALLOWED_ORIGINS
    allow_wildcard = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_wildcard else origins,
    allow_credentials=not allow_wildcard,  # RFC compliant: never allow credentials with wildcard origin
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "Retry-After",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Enforces essential defensive HTTP security headers on all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# --- 2. SLIDING WINDOW RATE LIMITING MIDDLEWARE ---
rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)
app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=rate_limiter,
    exempt_paths={"/health", "/docs", "/redoc", "/openapi.json", "/"},
)


# --- REQUEST & RESPONSE SCHEMAS ---
class TextIn(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The article text, press release, or headline claim to evaluate.",
        examples=["GENEVA (Reuters) - The World Health Organization confirmed today..."],
    )
    categories: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional category tag for auditing or logging purposes.",
    )
    bypass_cache: bool = Field(
        False,
        description="When true, bypasses the persistent prediction cache and forces fresh neural inference.",
    )

    @field_validator("text")
    @classmethod
    def sanitize_control_characters(cls, v: str) -> str:
        # Strip null bytes and unprintable terminal control characters
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", v)
        from News_Detection.ensemble import validate_claim_text
        return validate_claim_text(cleaned)


class UrlIn(BaseModel):
    url: str = Field(
        ...,
        min_length=4,
        max_length=2048,
        description="The full HTTP/HTTPS URL of the news article to scrape and analyze.",
        examples=["https://www.bbc.com/news/articles/c04j4l7j9pgo"],
    )
    categories: Optional[str] = Field(
        None,
        description="Optional category tag for auditing or logging purposes.",
    )

    @field_validator("url")
    @classmethod
    def validate_article_url(cls, v: str) -> str:
        from News_Detection.scraper import validate_target_url, InvalidURLError, SSRFBlockedError
        cleaned = v.strip()
        try:
            validate_target_url(cleaned)
        except (InvalidURLError, SSRFBlockedError) as err:
            raise ValueError(str(err))
        return cleaned



# --- 3. TIMEOUT & CONCURRENCY GUARD ---
async def execute_inference_safeguarded(
    text: str,
    categories: Optional[str] = None,
    bypass_cache: bool = False,
) -> Dict[str, Any]:
    """
    Executes CPU-bound neural network inference in a dedicated worker thread
    via asyncio.to_thread, bounded by concurrency semaphores and hard timeout limits.
    Serves instantaneous responses from the SHA-256 SQLite prediction cache if available.

    Guarantees:
      - Main event loop is never blocked (I/O, health checks, heartbeats remain instantaneous).
      - Server resources are protected against CPU thrashing via semaphore queuing.
      - Requests taking longer than INFERENCE_TIMEOUT_SECONDS return HTTP 504.
    """
    # 1. Fast-path Cache Hit (Bypasses Semaphore and Worker Thread Pool)
    if not bypass_cache:
        try:
            from News_Detection.storage import prediction_cache
            cached = prediction_cache.get(text, categories=categories, enable_grounding=True)
            if cached is not None:
                return cached
        except Exception as c_err:
            logger.debug(f"Cache check error in API safeguard: {c_err}")

    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)

    try:
        # Acquire slot without blocking the event loop
        async with _inference_semaphore:
            # Run CPU-bound ML forward passes in worker thread with timeout guard
            result = await asyncio.wait_for(
                asyncio.to_thread(analyze_text, text, categories),
                timeout=INFERENCE_TIMEOUT_SECONDS,
            )
            if not bypass_cache and isinstance(result, dict):
                try:
                    from News_Detection.storage import prediction_cache
                    prediction_cache.set(text, result, categories=categories, enable_grounding=True)
                except Exception:
                    pass
            return result

    except asyncio.TimeoutError:
        logger.error(f"Inference request timed out after {INFERENCE_TIMEOUT_SECONDS}s")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=(
                f"Inference timed out after {INFERENCE_TIMEOUT_SECONDS} seconds. "
                "The server was unable to complete the neural network analysis within the allotted window."
            ),
        )
    except ValueError as val_err:
        logger.warning(f"Invalid input: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception(f"Unexpected error during neural network inference: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during neural network analysis.",
        )


from News_Detection.accelerator import (
    get_hardware_status,
    get_runtime_environment,
    benchmark_acceleration,
)

# --- ROUTES ---
@app.get("/health", tags=["Diagnostics"])
async def health() -> Dict[str, Any]:
    """
    Lightweight health and readiness diagnostic.
    Exempt from rate limiting to guarantee monitoring probes always succeed.
    """
    return {
        "status": "ok",
        "models_loaded": are_models_loaded(),
        "models": [
            "RoBERTa Base (125M)",
            "Distilled TinyBERT (14M)",
            "BERT-Detect (110M)",
            "XLM-RoBERTa Multilingual (278M)"
        ],
        "multilingual_support": True,
        "web_grounding": True,
        "web_grounding_pool": "Persistent HTTP/1.1 Keep-Alive",
        "grounding_cache": "In-Memory LRU (10m TTL)",
        "article_scraper": True,
        "prediction_cache": True,
        "database": "SQLite (WAL Mode)",
        "probability_calibration": True,
        "temperature_scaling": True,
        "supported_categories": [
            "Politics",
            "Technology",
            "Business",
            "Sports",
            "Science",
            "Entertainment",
            "International News",
        ],
        "concurrency_limit": MAX_CONCURRENT_INFERENCES,
        "timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
        "runtime": get_runtime_environment(),
        "hardware_acceleration": get_hardware_status(),
    }



@app.get("/api/accelerator", tags=["Diagnostics"])
async def api_accelerator() -> Dict[str, Any]:
    """
    Returns real-time hardware accelerator device metadata, FP16 precision,
    and ONNX Runtime provider availability (MPS / CoreML / CUDA / CPU).
    """
    return get_hardware_status()


@app.get("/api/benchmark", tags=["Diagnostics"])
async def api_benchmark() -> Dict[str, Any]:
    """
    Runs a live hardware tensor execution benchmark comparing standard CPU
    latency with active accelerated hardware (e.g. Apple Silicon Metal FP16).
    """
    return benchmark_acceleration()


@app.get("/api/cache/stats", tags=["Diagnostics"])
async def api_cache_stats() -> Dict[str, Any]:
    """
    Returns persistent prediction cache metrics, active volume, hit ratios,
    and storage path.
    """
    from News_Detection.storage import prediction_cache
    return prediction_cache.get_stats()


@app.post("/api/cache/clear", tags=["Diagnostics"])
async def api_cache_clear(expired_only: bool = False) -> Dict[str, Any]:
    """
    Flushes expired entries (or entire cache if expired_only=False) from SQLite store.
    """
    from News_Detection.storage import prediction_cache
    if expired_only:
        cleared = prediction_cache.clear_expired()
        return {"status": "cleared", "mode": "expired_only", "entries_removed": cleared}
    cleared = prediction_cache.clear_all()
    return {"status": "cleared", "mode": "all", "entries_removed": cleared}


class CalibrationEvalIn(BaseModel):
    samples: List[Dict[str, Any]] = Field(
        ...,
        description="List of predictions e.g. [{'confidence': 0.95, 'prediction': 1, 'actual': 1}]",
    )
    num_bins: Optional[int] = Field(10, ge=2, le=50)


@app.get("/api/calibration/metrics", tags=["Diagnostics"])
async def api_calibration_metrics() -> Dict[str, Any]:
    """
    Returns active neural network temperature scaling parameters and calibration status.
    """
    from News_Detection.calibration import DEFAULT_TEMPERATURES
    return {
        "status": "enabled",
        "method": "Temperature Scaling (z_scaled = logit(p) / T)",
        "temperatures": DEFAULT_TEMPERATURES,
        "description": (
            "Applies post-processing temperature scaling to normalize overconfident "
            "transformer softmax distributions on out-of-distribution news claims."
        ),
    }


@app.post("/api/calibration/evaluate", tags=["Diagnostics"])
async def api_calibration_evaluate(payload: CalibrationEvalIn) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    Brier Score, and optimal temperature tuning for validation predictions.
    """
    from News_Detection.calibration import (
        compute_ece,
        compute_brier_score,
        compute_nll,
        optimize_temperature,
    )
    if not payload.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Samples list cannot be empty.",
        )

    confidences = [float(s.get("confidence", 0.5)) for s in payload.samples]
    predictions = [int(bool(s.get("prediction", 0))) for s in payload.samples]
    labels = [int(bool(s.get("actual", 0))) for s in payload.samples]

    ece_res = compute_ece(confidences, predictions, labels, num_bins=payload.num_bins or 10)
    brier = compute_brier_score(confidences, labels)
    nll = compute_nll(confidences, labels)
    opt = optimize_temperature(confidences, labels)

    return {
        "evaluation": ece_res,
        "brier_score": brier,
        "negative_log_likelihood": nll,
        "optimization": opt,
    }


@app.post("/api/analyze", tags=["Inference"])
async def api_analyze(payload: TextIn) -> Dict[str, Any]:
    """
    Analyzes news text using the 3-model Transformer ensemble with Soft Probability Consensus.
    
    Protected by:
      - Event-loop unblocking via asyncio.to_thread
      - Hard timeout safeguard (HTTP 504 Gateway Timeout)
      - Concurrency throttling semaphore (prevents CPU starvation)
      - IP-based Sliding Window Rate Limiting (HTTP 429 Too Many Requests)
      - Lexical entropy & repetition spam detection
    """
    from News_Detection.ensemble import validate_claim_text
    try:
        sanitized = validate_claim_text(payload.text)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    return await execute_inference_safeguarded(
        sanitized,
        payload.categories,
        bypass_cache=payload.bypass_cache,
    )


@app.post("/api/category", tags=["Inference"])
async def api_category(payload: TextIn) -> Dict[str, Any]:
    """
    Predicts the news category among the 7 standard taxonomies:
    Politics, Technology, Business, Sports, Science, Entertainment, International News.
    """
    sanitized = payload.text.strip()
    words = [w for w in sanitized.split() if w.strip()]
    if len(words) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Input contains only {len(words)} word{'s' if len(words) != 1 else ''}. "
                "News authenticity verification requires a complete claim, headline, or article body (minimum 3 words)."
            ),
        )
    from News_Detection.categorizer import predict_news_category
    return await asyncio.to_thread(predict_news_category, sanitized)
 
 
@app.post("/api/grounding", tags=["Fact-Checking"])
@app.post("/api/ground", tags=["Fact-Checking"], include_in_schema=False)
async def api_grounding(payload: TextIn) -> Dict[str, Any]:
    """
    Directly queries Google News RSS feeds across languages and Wikipedia
    to return fact-checking evidence, corroboration scores, and source links.
    """
    sanitized = payload.text.strip()
    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Claim text cannot be empty or contain only whitespace.",
        )
    from News_Detection.grounding import ground_claims_async
    from .ensemble import detect_language

    lang_code, _ = detect_language(sanitized)
    return await ground_claims_async(sanitized, lang_code=lang_code)


@app.get("/api/grounding/stats", tags=["Grounding"])
async def api_grounding_stats() -> Dict[str, Any]:
    """
    Returns in-ram web grounding LRU cache metrics, hit ratios, and connection pool parameters.
    """
    from News_Detection.grounding import grounding_cache
    return {
        "status": "enabled",
        "connection_pool": {
            "protocol": "HTTP/1.1 Persistent Keep-Alive",
            "max_keepalive": 20,
            "max_connections": 50,
            "keepalive_expiry_seconds": 60.0,
        },
        "cache": grounding_cache.get_stats(),
    }


@app.post("/api/scrape", tags=["Scraper"])
async def api_scrape(payload: UrlIn) -> Dict[str, Any]:
    """
    Scrapes and extracts clean article text, headline, author, and publisher
    metadata from any news URL with SSRF protection and boilerplate removal.
    """
    from News_Detection.scraper import (
        scrape_article,
        SSRFBlockedError,
        InvalidURLError,
        ContentExtractionError,
    )

    try:
        scraped = await asyncio.to_thread(scrape_article, payload.url)
        return scraped
    except SSRFBlockedError as ssrf_err:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(ssrf_err),
        )
    except InvalidURLError as url_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(url_err),
        )
    except ContentExtractionError as ext_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ext_err),
        )
    except Exception as exc:
        logger.exception(f"Unexpected error scraping URL {payload.url}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while scraping the article.",
        )


@app.post("/api/analyze-url", tags=["Inference"])
async def api_analyze_url(payload: UrlIn) -> Dict[str, Any]:
    """
    End-to-end Article URL Pipeline:
      1. Scrapes the article from the given URL with SSRF guards.
      2. Executes multi-model neural analysis with soft probability consensus.
      3. Cross-references live web grounding evidence and returns unified payload.
    """
    from News_Detection.scraper import (
        scrape_article,
        SSRFBlockedError,
        InvalidURLError,
        ContentExtractionError,
    )

    try:
        scraped = await asyncio.to_thread(scrape_article, payload.url)
    except SSRFBlockedError as ssrf_err:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(ssrf_err),
        )
    except InvalidURLError as url_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(url_err),
        )
    except ContentExtractionError as ext_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ext_err),
        )
    except Exception as exc:
        logger.exception(f"Unexpected error scraping URL {payload.url}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred while scraping the article.",
        )

    # Perform safeguarded neural inference on the extracted text
    analysis_result = await execute_inference_safeguarded(scraped["text"], payload.categories)

    # Attach rich article metadata
    analysis_result["article_meta"] = {
        "url": scraped["url"],
        "domain": scraped["domain"],
        "publisher": scraped["publisher"],
        "title": scraped["title"],
        "author": scraped["author"],
        "published_date": scraped["published_date"],
        "image_url": scraped["image_url"],
        "favicon": scraped["favicon"],
        "word_count": scraped["word_count"],
        "char_count": scraped["char_count"],
    }
    return analysis_result


async def read_file_bounded(file: UploadFile, max_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> bytes:
    """Reads uploaded file content in chunks up to max_bytes, raising HTTP 413 if exceeded."""
    chunk_size = 1024 * 1024  # 1MB chunks
    chunks = []
    total = 0
    http_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=http_413,
                detail=f"Uploaded file exceeds maximum allowed size of {max_bytes // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# --- BATCH PROCESSING ENDPOINTS (BATCH INGESTION) ---

@app.post("/api/batch/upload", tags=["Batch Processing"])
async def api_batch_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_column: Optional[str] = Form(None),
    enable_grounding: bool = Form(False),
) -> Dict[str, Any]:
    """
    Enterprise Batch Ingestion Endpoint:
      Accepts CSV, TSV, JSON, JSONL, TXT, or PDF files.
      Parses records, starts background multi-model classification,
      and returns a trackable job_id with initial metadata.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    content = await read_file_bounded(file)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        items, metadata = parse_batch_file(content, file.filename, text_column=text_column)
    except BatchProcessingError as b_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(b_err),
        )
    except Exception as exc:
        logger.exception(f"Failed to parse batch file {file.filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing document: {str(exc)}",
        )

    # Register job and dispatch background worker
    job_id = batch_job_manager.create_job(
        file.filename,
        items,
        metadata,
        enable_grounding=enable_grounding,
    )
    background_tasks.add_task(batch_job_manager.run_job_sync, job_id)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "total_items": len(items),
        "metadata": metadata,
    }


@app.get("/api/batch/jobs/{job_id}", tags=["Batch Processing"])
async def api_batch_get_job(job_id: str) -> Dict[str, Any]:
    """
    Fetches real-time status, progress percentage, summary analytics,
    and row-level classification results for a batch job.
    """
    job = batch_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found.",
        )
    return job


@app.get("/api/batch/jobs", tags=["Batch Processing"])
async def api_batch_list_jobs() -> Dict[str, Any]:
    """
    Lists recent batch processing jobs and their summary metrics.
    """
    jobs = batch_job_manager.list_jobs()
    return {"jobs": jobs, "total_jobs": len(jobs)}


@app.get("/api/batch/jobs/{job_id}/export", tags=["Batch Processing"])
async def api_batch_export(job_id: str, format: str = "csv") -> Any:
    """
    Exports completed batch job results into enriched CSV or JSON format.
    """
    job = batch_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{job_id}' not found.",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is in '{job['status']}' status and cannot be exported yet.",
        )

    if format.lower() == "json":
        return job

    try:
        csv_content = batch_job_manager.export_csv(job_id)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=batch_audit_{job_id}.csv",
                "Cache-Control": "no-cache",
            },
        )
    except Exception as exc:
        logger.exception(f"Failed exporting batch job {job_id} to CSV: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("News_Detection.health_api:app", host="127.0.0.1", port=8000, reload=True)