"""
Unit tests for Article URL Scraper & Web Crawler Engine (Phase 5).
Validates URL validation, SSRF security guards, metadata extraction,
boilerplate removal, and paywall error handling.
"""

import unittest
import urllib.parse
from unittest.mock import patch
from bs4 import BeautifulSoup
from News_Detection.scraper import (
    validate_target_url,
    clean_article_text,
    extract_metadata,
    extract_article_body,
    scrape_article,
    SafeRedirectHandler,
    SSRFBlockedError,
    InvalidURLError,
    ContentExtractionError,
)


SAMPLE_NEWS_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Global Leaders Sign Historic AI Safety Accord - TechWire</title>
    <meta property="og:title" content="Global Leaders Sign Historic AI Safety Accord" />
    <meta property="og:site_name" content="TechWire Global" />
    <meta property="article:author" content="Elena Vance" />
    <meta property="article:published_time" content="2026-10-25T14:30:00Z" />
    <meta property="og:image" content="https://techwire.example.com/images/summit.jpg" />
</head>
<body>
    <header>
        <nav><a href="/">Home</a><a href="/news">News</a></nav>
    </header>
    <div class="cookie-banner">Please accept cookies to proceed.</div>
    <div class="advertisement">Buy new smartphones today!</div>
    <article>
        <h1>Global Leaders Sign Historic AI Safety Accord</h1>
        <p class="byline">By Elena Vance | TechWire</p>
        <p>GENEVA — In an unprecedented diplomatic breakthrough, representatives from over forty sovereign nations concluded negotiations today on international safeguards governing artificial intelligence architectures.</p>
        <p>The multilateral treaty establishes rigorous independent evaluation protocols for autonomous decision systems operating across critical public infrastructure and financial systems.</p>
        <p>Delegates confirmed that the oversight framework emphasizes transparency, red-teaming audits, and open vulnerability disclosures across all participating institutions.</p>
    </article>
    <div class="sidebar">Related: Top Tech Trends of 2026</div>
    <footer>Copyright 2026 TechWire. All rights reserved.</footer>
</body>
</html>
"""


class TestArticleScraper(unittest.TestCase):
    @patch("socket.getaddrinfo")
    def test_validate_target_url_valid_https(self, mock_addrinfo):
        mock_addrinfo.return_value = [(None, None, None, None, ("151.101.1.67", 443))]
        parsed = validate_target_url("https://www.reuters.com/world/article-1")
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "www.reuters.com")

    @patch("socket.getaddrinfo")
    def test_validate_target_url_auto_prefix_http(self, mock_addrinfo):
        mock_addrinfo.return_value = [(None, None, None, None, ("151.101.1.67", 443))]
        parsed = validate_target_url("bbc.com/news/world")
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.hostname, "bbc.com")

    def test_validate_target_url_invalid_schemes(self):
        with self.assertRaises(InvalidURLError):
            validate_target_url("ftp://example.com/file.txt")
        with self.assertRaises(InvalidURLError):
            validate_target_url("file:///etc/passwd")

    def test_validate_target_url_blocks_forbidden_ports(self):
        # Disallow non-standard / dangerous service ports
        with self.assertRaises(InvalidURLError):
            validate_target_url("http://example.com:22/secret")  # SSH
        with self.assertRaises(InvalidURLError):
            validate_target_url("http://example.com:6379/")      # Redis
        with self.assertRaises(InvalidURLError):
            validate_target_url("http://example.com:3306/")      # MySQL

    def test_validate_target_url_blocks_userinfo_credentials(self):
        with self.assertRaises(InvalidURLError):
            validate_target_url("http://admin:secret@example.com/article")
        with self.assertRaises(InvalidURLError):
            validate_target_url("https://user@reuters.com/news")

    def test_validate_target_url_blocks_social_media_platforms(self):
        """Phase 6.2: Verifies non-article social media links raise InvalidURLError."""
        for social in [
            "https://twitter.com/user/status/12345",
            "https://x.com/news/status/6789",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://instagram.com/p/C_12345",
            "https://tiktok.com/@creator/video/123",
        ]:
            with self.assertRaises(InvalidURLError) as ctx:
                validate_target_url(social)
            self.assertIn("cannot be scraped as standard news articles", str(ctx.exception))

    @patch("News_Detection.scraper.fetch_article_html")
    @patch("socket.getaddrinfo")
    def test_scrape_article_detects_paywall_markers(self, mock_addrinfo, mock_fetch):
        """Phase 6.2: Verifies paywalled pages raise ContentExtractionError."""
        mock_addrinfo.return_value = [(None, None, None, None, ("151.101.1.67", 443))]
        paywall_html = """
        <html>
            <head><title>Exclusive Market Analysis</title></head>
            <body>
                <div class="paywall-overlay">
                    <h2>This article is subscriber-only</h2>
                    <p>Subscribe now to read the full story.</p>
                </div>
            </body>
        </html>
        """
        mock_fetch.return_value = paywall_html
        with self.assertRaises(ContentExtractionError) as ctx:
            scrape_article("https://www.ft.com/content/123")
        self.assertIn("Paywalled or subscription-gated article detected", str(ctx.exception))

    def test_ssrf_protection_blocks_localhost_and_loopback(self):
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://localhost:8000/api/analyze")
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://127.0.0.1:3000")
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://0.0.0.0:8000")

    @patch("socket.getaddrinfo")
    def test_ssrf_protection_blocks_private_subnets(self, mock_addrinfo):
        # Mock DNS resolution returning private IPv4 address
        mock_addrinfo.return_value = [
            (None, None, None, None, ("192.168.1.100", 80))
        ]
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://internal-corp-gateway.com")

        # Mock AWS EC2 metadata link-local address
        mock_addrinfo.return_value = [
            (None, None, None, None, ("169.254.169.254", 80))
        ]
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://cloud-instance-meta.com")

    @patch("socket.getaddrinfo")
    def test_ssrf_protection_blocks_ipv4_mapped_ipv6(self, mock_addrinfo):
        # IPv4-mapped IPv6 loopback (::ffff:127.0.0.1)
        mock_addrinfo.return_value = [
            (None, None, None, None, ("::ffff:127.0.0.1", 80))
        ]
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://spoofed-ipv6-loopback.com")

        # IPv4-mapped IPv6 link-local (::ffff:169.254.169.254)
        mock_addrinfo.return_value = [
            (None, None, None, None, ("::ffff:169.254.169.254", 80))
        ]
        with self.assertRaises(SSRFBlockedError):
            validate_target_url("http://spoofed-ipv6-metadata.com")

    def test_safe_redirect_handler_blocks_private_destinations(self):
        handler = SafeRedirectHandler()
        # Direct redirect to localhost
        with self.assertRaises(SSRFBlockedError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1:8000/api/analyze")

        with self.assertRaises(SSRFBlockedError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://localhost:3000")

    @patch("socket.getaddrinfo")
    def test_safe_redirect_handler_blocks_excessive_redirects(self, mock_addrinfo):
        mock_addrinfo.return_value = [(None, None, None, None, ("151.101.1.67", 443))]
        handler = SafeRedirectHandler(max_redirects=2)
        base_req = urllib.request.Request("https://example.com/initial")
        # 1st redirect (valid external URL)
        req1 = handler.redirect_request(base_req, None, 302, "Found", {}, "https://reuters.com/news")
        self.assertIsNotNone(req1)
        # 2nd redirect
        req2 = handler.redirect_request(base_req, None, 302, "Found", {}, "https://reuters.com/world")
        self.assertIsNotNone(req2)
        # 3rd redirect exceeds limit
        with self.assertRaises(SSRFBlockedError):
            handler.redirect_request(base_req, None, 302, "Found", {}, "https://reuters.com/article")

    def test_extract_metadata(self):
        soup = BeautifulSoup(SAMPLE_NEWS_HTML, "html.parser")
        parsed_url = urllib.parse.urlparse("https://techwire.example.com/article/ai-safety")
        meta = extract_metadata(soup, parsed_url)

        self.assertEqual(meta["title"], "Global Leaders Sign Historic AI Safety Accord")
        self.assertEqual(meta["publisher"], "TechWire Global")
        self.assertEqual(meta["author"], "Elena Vance")
        self.assertEqual(meta["published_date"], "2026-10-25T14:30:00Z")
        self.assertEqual(meta["image_url"], "https://techwire.example.com/images/summit.jpg")

    def test_extract_article_body_strips_boilerplate(self):
        soup = BeautifulSoup(SAMPLE_NEWS_HTML, "html.parser")
        body_text = extract_article_body(soup)

        self.assertNotIn("Please accept cookies", body_text)
        self.assertNotIn("Buy new smartphones", body_text)
        self.assertNotIn("Copyright 2026 TechWire", body_text)
        self.assertIn("GENEVA — In an unprecedented diplomatic breakthrough", body_text)
        self.assertIn("The multilateral treaty establishes rigorous", body_text)
        self.assertIn("Delegates confirmed that the oversight framework", body_text)

    def test_clean_article_text(self):
        dirty = "  First line.   \r\n\r\n\n  Second line with   spaces.  \n\n\n\nThird line. "
        cleaned = clean_article_text(dirty)
        expected = "First line.\n\nSecond line with spaces.\n\nThird line."
        self.assertEqual(cleaned, expected)

    @patch("socket.getaddrinfo")
    @patch("News_Detection.scraper.fetch_article_html")
    def test_scrape_article_end_to_end_mock(self, mock_fetch_html, mock_addrinfo):
        mock_addrinfo.return_value = [
            (None, None, None, None, ("93.184.216.34", 80))
        ]
        mock_fetch_html.return_value = SAMPLE_NEWS_HTML

        res = scrape_article("https://techwire.example.com/article/ai-safety")

        self.assertEqual(res["title"], "Global Leaders Sign Historic AI Safety Accord")
        self.assertEqual(res["publisher"], "TechWire Global")
        self.assertGreater(res["word_count"], 40)
        self.assertIn("techwire.example.com", res["domain"])
        self.assertIn("GENEVA", res["text"])



if __name__ == "__main__":
    unittest.main()
