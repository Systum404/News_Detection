import unittest
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from News_Detection.rate_limiter import SlidingWindowRateLimiter, resolve_client_ip

try:
    from fastapi import HTTPException
    from News_Detection import health_api
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    HTTPException = object  # type: ignore
    health_api = None  # type: ignore


class TestRateLimiter(unittest.TestCase):

    def setUp(self):
        # Create rate limiter allowing 3 requests per 2 seconds for testing
        self.limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=2)

    def test_rate_limiter_allows_up_to_max(self):
        client = "192.168.1.10"
        # 1st request
        allowed, remaining, retry_after = self.limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 2)
        self.assertEqual(retry_after, 0)

        # 2nd request
        allowed, remaining, retry_after = self.limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 1)

        # 3rd request (max reached)
        allowed, remaining, retry_after = self.limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 0)

        # 4th request (should be rejected)
        allowed, remaining, retry_after = self.limiter.is_allowed(client)
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(retry_after, 0)

    def test_rate_limiter_client_isolation(self):
        client_a = "10.0.0.1"
        client_b = "10.0.0.2"

        # Exhaust client A
        for _ in range(3):
            self.limiter.is_allowed(client_a)

        self.assertFalse(self.limiter.is_allowed(client_a)[0])

        # Client B must remain unaffected
        allowed_b, remaining_b, _ = self.limiter.is_allowed(client_b)
        self.assertTrue(allowed_b)
        self.assertEqual(remaining_b, 2)

    def test_rate_limiter_window_expiration(self):
        client = "172.16.0.5"
        # Exhaust limit
        for _ in range(3):
            self.limiter.is_allowed(client)
        self.assertFalse(self.limiter.is_allowed(client)[0])

        # Simulate time passing beyond window
        time.sleep(2.1)

        # Should be allowed again
        allowed, remaining, _ = self.limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 2)

    def test_resolve_client_ip_blocks_spoofing_from_untrusted_peer(self):
        # Direct untrusted client attempting spoofing via X-Forwarded-For
        mock_req = MagicMock()
        mock_req.client.host = "203.0.113.55"  # External peer
        mock_req.headers = {"X-Forwarded-For": "8.8.8.8, 1.1.1.1"}

        # With default trusted proxies (127.0.0.1, ::1, localhost), 203.0.113.55 is NOT trusted
        resolved = resolve_client_ip(mock_req, trusted_proxies={"127.0.0.1", "::1", "localhost"})
        self.assertEqual(resolved, "203.0.113.55")  # Spoofed header must be ignored!

    def test_resolve_client_ip_respects_trusted_reverse_proxy(self):
        # Request forwarded by trusted reverse proxy (e.g. Next.js / Nginx on 127.0.0.1)
        mock_req = MagicMock()
        mock_req.client.host = "127.0.0.1"
        mock_req.headers = {"X-Forwarded-For": "198.51.100.42"}

        resolved = resolve_client_ip(mock_req, trusted_proxies={"127.0.0.1", "::1", "localhost"})
        self.assertEqual(resolved, "198.51.100.42")


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is not installed in current python environment")
class TestApiSafeguards(unittest.TestCase):

    def setUp(self):
        # Set short timeout for testing
        self.original_timeout = health_api.INFERENCE_TIMEOUT_SECONDS
        health_api.INFERENCE_TIMEOUT_SECONDS = 0.2

    def tearDown(self):
        health_api.INFERENCE_TIMEOUT_SECONDS = self.original_timeout

    def test_inference_timeout_triggers_504(self):
        """Tests that slow ML inference is aborted and cleanly raises HTTP 504 Gateway Timeout."""
        def slow_mock_inference(text, categories=None):
            time.sleep(0.5)  # Longer than 0.2s timeout
            return {"result": "AUTHENTIC"}

        with patch("News_Detection.health_api.analyze_text", side_effect=slow_mock_inference):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with self.assertRaises(HTTPException) as ctx:
                    loop.run_until_complete(
                        health_api.execute_inference_safeguarded("Test news article headline")
                    )
                self.assertEqual(ctx.exception.status_code, 504)
                self.assertIn("timed out", ctx.exception.detail.lower())
            finally:
                loop.close()

    def test_inference_successful_under_timeout(self):
        """Tests that fast inference completes normally and returns results."""
        expected_result = {
            "result": "AUTHENTIC NEWS",
            "verdict": "AUTHENTIC",
            "is_fake": False,
            "confidence": "95.0% (3/3 Agree)",
        }

        def fast_mock_inference(text, categories=None):
            return expected_result

        with patch("News_Detection.health_api.analyze_text", side_effect=fast_mock_inference):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res = loop.run_until_complete(
                    health_api.execute_inference_safeguarded("Valid news article headline")
                )
                self.assertEqual(res["result"], "AUTHENTIC NEWS")
                self.assertEqual(res["verdict"], "AUTHENTIC")
            finally:
                loop.close()

    def test_inference_returns_sha256_and_consensus_nuance(self):
        """Verifies execute_inference_safeguarded propagates claim_sha256 and is_polarized."""
        mock_result = {
            "result": "UNCONFIRMED // HIGH MODEL POLARIZATION",
            "verdict": "UNCONFIRMED",
            "is_fake": False,
            "is_polarized": True,
            "polarization_index": 0.02,
            "claim_sha256": "abcdef0123456789" * 4,
            "confidence": "51.0% (2/3 Agree)",
        }
        with patch("News_Detection.health_api.analyze_text", return_value=mock_result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                res = loop.run_until_complete(
                    health_api.execute_inference_safeguarded("Valid news article headline", bypass_cache=True)
                )
                self.assertEqual(res["verdict"], "UNCONFIRMED")
                self.assertTrue(res["is_polarized"])
                self.assertEqual(len(res["claim_sha256"]), 64)
            finally:
                loop.close()

    def test_empty_or_whitespace_text_rejected(self):
        """Tests that empty or whitespace text is rejected during validation."""
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, ValueError)):
            health_api.TextIn(text="   ")

    def test_single_word_and_short_non_claim_rejected(self):
        """Tests that single word (e.g. 'System') or 2-word inputs are rejected with validation error."""
        from pydantic import ValidationError
        for invalid in ["System", "Apple", "Breaking news", "Just headline"]:
            with self.assertRaises((ValidationError, ValueError)):
                health_api.TextIn(text=invalid)

    def test_repetition_spam_rejected_in_schema(self):
        """Phase 6.3: Verifies repetitive token spam and low-entropy inputs are rejected by TextIn."""
        from pydantic import ValidationError
        for spam in [
            "System System System",
            "Fake news! Fake news! Fake news! Fake news!",
        ]:
            with self.assertRaises((ValidationError, ValueError)):
                health_api.TextIn(text=spam)

    def test_social_media_url_rejected_in_schema(self):
        """Phase 6.2: Verifies social media and streaming URLs are rejected by UrlIn."""
        from pydantic import ValidationError
        for url in [
            "https://twitter.com/user/status/123",
            "https://x.com/post/456",
            "https://youtube.com/watch?v=789",
        ]:
            with self.assertRaises((ValidationError, ValueError)):
                health_api.UrlIn(url=url)

    def test_read_file_bounded_blocks_oversized_upload(self):
        """Tests that uploads exceeding max_bytes raise HTTP 413 Payload Too Large."""
        mock_file = MagicMock()
        # Simulate a file yielding 2 chunks of 100 bytes when max is 150
        mock_file.read = AsyncMock(side_effect=[b"a" * 100, b"b" * 100, b""])

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with self.assertRaises(HTTPException) as ctx:
                loop.run_until_complete(health_api.read_file_bounded(mock_file, max_bytes=150))
            self.assertEqual(ctx.exception.status_code, 413)
            self.assertIn("exceeds maximum allowed size", ctx.exception.detail.lower())
        finally:
            loop.close()

    def test_text_in_strips_null_bytes_and_control_chars(self):
        """Tests that null bytes and unprintable terminal control characters are stripped."""
        payload = health_api.TextIn(text="Breaking news\x00\x07 headline\x08 statement")
        self.assertEqual(payload.text, "Breaking news headline statement")

    def test_security_headers_present_on_responses(self):
        """Tests that HTTP responses include essential security headers."""
        from starlette.testclient import TestClient
        client = TestClient(health_api.app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(resp.headers.get("x-frame-options"), "DENY")
        self.assertEqual(resp.headers.get("x-xss-protection"), "1; mode=block")
        self.assertEqual(resp.headers.get("referrer-policy"), "strict-origin-when-cross-origin")


if __name__ == "__main__":
    unittest.main()
