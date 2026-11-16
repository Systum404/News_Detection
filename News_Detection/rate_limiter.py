import os
import time
import threading
from collections import deque
from typing import Dict, Tuple, Optional, Set, Any
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response, JSONResponse
except ImportError:
    BaseHTTPMiddleware = object
    Request = Any  # type: ignore
    Response = Any  # type: ignore
    JSONResponse = Any  # type: ignore


def resolve_client_ip(request: Request, trusted_proxies: Optional[Set[str]] = None) -> str:
    """
    Safely resolves the true client IP address.

    Security:
      - Ignores X-Forwarded-For if request.client.host is not in trusted_proxies.
      - Prevents attackers from sending spoofed X-Forwarded-For headers to evade rate limits.
      - When behind trusted reverse proxies, takes the closest untrusted client IP from right to left.
    """
    direct_ip = request.client.host if (request.client and request.client.host) else "unknown"
    if not trusted_proxies:
        return direct_ip

    if direct_ip in trusted_proxies:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            for hop in reversed(hops):
                if hop not in trusted_proxies:
                    return hop
            if hops:
                return hops[0]
    return direct_ip


class SlidingWindowRateLimiter:
    """
    Thread-safe, in-memory sliding window rate limiter.

    Tracks request timestamps per client IP within a rolling time window.
    Prunes expired entries to ensure minimal memory overhead.
    """

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: int = 60,
        cleanup_interval: int = 300,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        self._requests: Dict[str, deque] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Evaluates whether a request from client_id is permitted.

        Returns:
            Tuple[bool, int, int]: (allowed, remaining_requests, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Periodic cleanup of inactive client histories
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired(now)

            client_history = self._requests.setdefault(client_id, deque())

            # Evict timestamps outside the active sliding window
            while client_history and client_history[0] <= window_start:
                client_history.popleft()

            current_count = len(client_history)

            if current_count < self.max_requests:
                client_history.append(now)
                remaining = self.max_requests - (current_count + 1)
                return True, remaining, 0
            else:
                oldest_timestamp = client_history[0]
                retry_after = max(1, int(oldest_timestamp + self.window_seconds - now))
                return False, 0, retry_after

    def _cleanup_expired(self, now: float) -> None:
        """Prunes inactive client histories."""
        cutoff = now - self.window_seconds
        inactive_keys = [
            cid for cid, history in self._requests.items()
            if not history or history[-1] <= cutoff
        ]
        for cid in inactive_keys:
            del self._requests[cid]
        self._last_cleanup = now

    def reset(self) -> None:
        """Resets all tracked requests (primarily for test environments)."""
        with self._lock:
            self._requests.clear()
            self._last_cleanup = time.time()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI / Starlette middleware enforcing IP-based sliding window rate limits.
    
    Includes automatic exemption for health checks and OpenAPI docs,
    and returns RFC-compliant 429 Too Many Requests responses with Retry-After headers.
    """

    def __init__(
        self,
        app,
        rate_limiter: SlidingWindowRateLimiter,
        exempt_paths: Optional[Set[str]] = None,
        trusted_proxies: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/redoc", "/openapi.json", "/"}
        if trusted_proxies is None:
            env_proxies = os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1,localhost")
            self.trusted_proxies = {p.strip() for p in env_proxies.split(",") if p.strip()}
        else:
            self.trusted_proxies = trusted_proxies

    async def dispatch(self, request: Request, call_next) -> Response:
        # Bypass rate limiting for monitoring, documentation, and batch status endpoints
        if request.url.path in self.exempt_paths or request.url.path.startswith("/api/batch/jobs"):
            return await call_next(request)

        # Safely determine true client IP preventing spoofing
        client_ip = resolve_client_ip(request, self.trusted_proxies)

        allowed, remaining, retry_after = self.rate_limiter.is_allowed(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded. Maximum {self.rate_limiter.max_requests} requests "
                        f"per {self.rate_limiter.window_seconds} seconds. Please retry in {retry_after}s."
                    )
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.rate_limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
