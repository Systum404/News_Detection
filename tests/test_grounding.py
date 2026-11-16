"""
Unit tests for Real-Time Fact-Checking & Web Grounding Engine (RAG).
Validates query sanitization, feed fetching, corroboration metrics, connection pooling,
in-ram grounding caching, and async execution.
"""

import asyncio
import time
import unittest
from unittest.mock import patch
from News_Detection.grounding import (
    clean_search_query,
    extract_temporal_markers,
    evaluate_wikipedia_historical_grounding,
    ground_claims,
    ground_claims_async,
    get_http_client,
    grounding_cache,
    parse_rss_xml,
    parse_wikipedia_json,
    KNOWN_AUTHORITATIVE_PUBLISHERS,
)
from News_Detection.ensemble import analyze_text


class TestFactCheckingGrounding(unittest.TestCase):
    def setUp(self):
        grounding_cache.clear()

    def test_clean_search_query_dateline_removal(self):
        sample = "WASHINGTON (Reuters) - Federal Reserve Chair Jerome Powell announced interest rates will remain unchanged."
        cleaned = clean_search_query(sample)
        self.assertNotIn("Reuters", cleaned)
        self.assertNotIn("WASHINGTON", cleaned)
        self.assertTrue(cleaned.startswith("Federal Reserve Chair Jerome Powell"))

    def test_clean_search_query_clickbait_removal(self):
        sample = "BREAKING: Scientists discover revolutionary quantum computing superconductor in laboratory experiment!"
        cleaned = clean_search_query(sample)
        self.assertFalse(cleaned.lower().startswith("breaking"))
        self.assertTrue("Scientists discover revolutionary" in cleaned)

    def test_clean_search_query_hindi(self):
        sample = "नई दिल्ली: भारतीय रिजर्व बैंक ने मौद्रिक नीति समिति की बैठक में रेपो रेट को अपरिवर्तित रखा है।"
        cleaned = clean_search_query(sample)
        self.assertNotIn("नई दिल्ली", cleaned)
        self.assertTrue("भारतीय रिजर्व बैंक" in cleaned)

    def test_ground_claims_empty_or_short(self):
        res = ground_claims("hi")
        self.assertEqual(res["corroboration_status"], "INSUFFICIENT_INPUT")
        self.assertEqual(res["corroboration_score"], 0.0)
        self.assertEqual(len(res["sources"]), 0)

    @patch("News_Detection.grounding.fetch_news_rss")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding")
    def test_ground_claims_unverified_claim(self, mock_wiki, mock_rss):
        mock_rss.return_value = []
        mock_wiki.return_value = []

        res = ground_claims("Aliens from Alpha Centauri landed on the White House lawn yesterday.")
        self.assertEqual(res["corroboration_status"], "UNVERIFIED_CLAIM")
        self.assertEqual(res["corroboration_score"], 5.0)
        self.assertEqual(res["authoritative_count"], 0)
        self.assertIn("No corroborating reporting", res["summary"])

    @patch("News_Detection.grounding.fetch_news_rss")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding")
    def test_ground_claims_highly_corroborated(self, mock_wiki, mock_rss):
        mock_rss.return_value = [
            {
                "title": "Federal Reserve Holds Rates Steady",
                "publisher": "Reuters",
                "url": "https://reuters.com/rates",
                "published_at": "Sun, 25 Oct 2026",
                "is_authoritative": True,
                "type": "news_article",
            },
            {
                "title": "Fed leaves benchmark interest rate unchanged",
                "publisher": "Associated Press",
                "url": "https://apnews.com/fed",
                "published_at": "Sun, 25 Oct 2026",
                "is_authoritative": True,
                "type": "news_article",
            },
        ]
        mock_wiki.return_value = []

        res = ground_claims("Federal Reserve keeps interest rates steady in monetary policy meeting.")
        self.assertEqual(res["corroboration_status"], "HIGHLY_CORROBORATED")
        self.assertGreaterEqual(res["corroboration_score"], 85.0)
        self.assertEqual(res["authoritative_count"], 2)
        self.assertEqual(len(res["sources"]), 2)

    def test_analyze_text_includes_fact_check_payload(self):
        # When mock pipes are passed and enable_grounding is omitted, grounding is skipped to maintain >500 ops/sec
        mock_pipe = lambda t, **kwargs: [{"label": "REAL", "score": 0.95}]
        pipes = (mock_pipe, mock_pipe, mock_pipe)
        result = analyze_text("NASA launches new deep space telescope into lunar orbit.", pipes=pipes)

        self.assertIn("fact_check", result)
        self.assertEqual(result["fact_check"]["corroboration_status"], "SKIPPED")

    @patch("News_Detection.grounding.fetch_news_rss")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding")
    def test_concurrent_grounding_in_analyze_text(self, mock_wiki, mock_rss):
        # When enable_grounding=True is explicitly passed, grounding runs concurrently alongside neural inference
        mock_rss.return_value = [
            {
                "title": "NASA James Webb Space Telescope discovers ancient galaxy",
                "publisher": "Reuters",
                "url": "https://reuters.com/jwst",
                "published_at": "Today",
                "is_authoritative": True,
                "type": "news_article",
            }
        ]
        mock_wiki.return_value = []
        mock_pipe = lambda t, **kwargs: [{"label": "REAL", "score": 0.98}]
        pipes = (mock_pipe, mock_pipe, mock_pipe)

        result = analyze_text(
            "NASA James Webb Space Telescope discovers ancient galaxy at cosmic dawn.",
            pipes=pipes,
            enable_grounding=True,
        )

        self.assertIn("fact_check", result)
        self.assertNotEqual(result["fact_check"]["corroboration_status"], "SKIPPED")
        self.assertEqual(result["fact_check"]["authoritative_count"], 1)

    def test_authoritative_publishers_registry(self):
        self.assertIn("reuters", KNOWN_AUTHORITATIVE_PUBLISHERS)
        self.assertIn("bbc", KNOWN_AUTHORITATIVE_PUBLISHERS)
        self.assertIn("the hindu", KNOWN_AUTHORITATIVE_PUBLISHERS)
        self.assertIn("pib", KNOWN_AUTHORITATIVE_PUBLISHERS)

    def test_connection_pooling_singleton(self):
        client1 = get_http_client()
        client2 = get_http_client()
        self.assertIs(client1, client2)
        self.assertFalse(client1.is_closed)

    def test_in_memory_grounding_cache_performance(self):
        claim = "European Central Bank announces quarter point interest rate cut."
        with patch("News_Detection.grounding.fetch_news_rss") as mock_rss, \
             patch("News_Detection.grounding.fetch_wikipedia_grounding") as mock_wiki:
            mock_rss.return_value = [
                {
                    "title": "European Central Bank cuts interest rates",
                    "publisher": "Reuters",
                    "url": "https://reuters.com/ecb",
                    "published_at": "Today",
                    "is_authoritative": True,
                    "type": "news_article",
                }
            ]
            mock_wiki.return_value = []

            # 1. Cold call: cache miss
            t0 = time.perf_counter()
            first_res = ground_claims(claim, use_cache=True)
            t_cold = time.perf_counter() - t0
            self.assertEqual(mock_rss.call_count, 1)
            self.assertFalse(first_res.get("cached", False))

            # 2. Warm call: cache hit (sub-millisecond)
            t1 = time.perf_counter()
            second_res = ground_claims(claim, use_cache=True)
            t_warm = time.perf_counter() - t1
            self.assertEqual(mock_rss.call_count, 1)  # No second network call!
            self.assertTrue(second_res.get("cached", False))
            self.assertLess(t_warm, 0.005)  # Under 5ms (usually <0.1ms)

            # Check cache stats
            stats = grounding_cache.get_stats()
            self.assertEqual(stats["hits"], 1)
            self.assertEqual(stats["misses"], 1)
            self.assertGreater(stats["hit_ratio_pct"], 0)

    @patch("News_Detection.grounding.fetch_news_rss_async")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding_async")
    def test_ground_claims_async(self, mock_wiki_async, mock_rss_async):
        async def run_test():
            mock_rss_async.return_value = [
                {
                    "title": "Quantum supremacy demonstrated by research consortium",
                    "publisher": "Bloomberg",
                    "url": "https://bloomberg.com/quantum",
                    "published_at": "Today",
                    "is_authoritative": True,
                    "type": "news_article",
                }
            ]
            mock_wiki_async.return_value = []
            res = await ground_claims_async("Quantum supremacy demonstrated by major research lab consortium.", use_cache=False)
            self.assertEqual(res["corroboration_status"], "PARTIALLY_CORROBORATED")
            self.assertEqual(res["authoritative_count"], 1)
            self.assertEqual(len(res["sources"]), 1)

        asyncio.run(run_test())

    def test_parse_rss_xml_valid(self):
        sample_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Google News</title>
                <item>
                    <title>Stock Market Rallies After Earnings - Bloomberg</title>
                    <link>https://bloomberg.com/stocks</link>
                    <pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate>
                    <source url="https://bloomberg.com">Bloomberg</source>
                </item>
            </channel>
        </rss>"""
        parsed = parse_rss_xml(sample_xml, max_results=5)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["title"], "Stock Market Rallies After Earnings")
        self.assertEqual(parsed[0]["publisher"], "Bloomberg")
        self.assertTrue(parsed[0]["is_authoritative"])

    def test_parse_wikipedia_json_valid(self):
        sample_json = b"""{
            "query": {
                "search": [
                    {
                        "title": "Artificial general intelligence",
                        "snippet": "Artificial general intelligence is the <span class=\\"searchmatch\\">intelligence</span> of a machine."
                    }
                ]
            }
        }"""
        parsed = parse_wikipedia_json(sample_json, max_results=2)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["title"], "Artificial general intelligence")
        self.assertEqual(parsed[0]["publisher"], "Wikipedia")
        self.assertNotIn("<span", parsed[0]["snippet"])

    def test_extract_temporal_markers(self):
        """Milestone 8.1: Tests extraction of historical years and epochal markers."""
        # Past historical year
        m1 = extract_temporal_markers("In 1969 Apollo 11 landed on the Moon.")
        self.assertTrue(m1["is_historical"])
        self.assertIn(1969, m1["historical_years"])

        # Multiple historical years
        m2 = extract_temporal_markers("World War II lasted from 1939 to 1945.")
        self.assertTrue(m2["is_historical"])
        self.assertEqual(m2["historical_years"], [1939, 1945])

        # Historical cue without explicit 4-digit year
        m3 = extract_temporal_markers("The treaty signed during the Cold War changed global diplomacy.")
        self.assertTrue(m3["is_historical"])
        self.assertTrue(m3["has_historical_cues"])

        # Modern claim without past years
        m4 = extract_temporal_markers("Federal Reserve holds interest rate meeting today.")
        self.assertFalse(m4["is_historical"])
        self.assertEqual(m4["historical_years"], [])

    @patch("News_Detection.grounding.fetch_news_rss")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding")
    def test_historical_fact_grounding_via_wikipedia(self, mock_wiki, mock_rss):
        """
        Milestone 8.1 & 8.2: Tests that an undisputed historical fact with 0 live wire
        RSS matches is corroborated by the Wikipedia Knowledge Graph.
        """
        # Google News RSS returns 0 matches because the event is decades old
        mock_rss.return_value = []
        # Wikipedia Knowledge Graph returns encyclopedic record confirming event & year
        mock_wiki.return_value = [
            {
                "title": "Apollo 11",
                "publisher": "Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Apollo_11",
                "snippet": "Apollo 11 was the spaceflight that landed Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969.",
                "published_at": "Encyclopedic Record",
                "is_authoritative": True,
                "type": "encyclopedia",
            }
        ]

        claim = "Apollo 11 astronaut Neil Armstrong walked on the Moon in 1969."
        res = ground_claims(claim, use_cache=False)

        self.assertEqual(res["corroboration_status"], "HISTORICAL_FACT_GROUNDED")
        self.assertGreaterEqual(res["corroboration_score"], 90.0)
        self.assertTrue(res["is_historical"])
        self.assertIsNotNone(res["temporal_metadata"])
        self.assertTrue(res["temporal_metadata"]["historical_fact"])
        self.assertIn(1969, res["temporal_metadata"]["verified_years"])
        self.assertIn("Wikipedia Knowledge Graph", res["summary"])

    @patch("News_Detection.grounding.fetch_news_rss_async")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding_async")
    def test_historical_fact_grounding_async(self, mock_wiki_async, mock_rss_async):
        """Milestone 8.1: Tests asynchronous historical grounding flow."""
        async def run_test():
            mock_rss_async.return_value = []
            mock_wiki_async.return_value = [
                {
                    "title": "Barack Obama",
                    "publisher": "Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Barack_Obama",
                    "snippet": "Barack Obama served as the 44th president of the United States after being elected in 2008.",
                    "published_at": "Encyclopedic Record",
                    "is_authoritative": True,
                    "type": "encyclopedia",
                }
            ]
            claim = "Barack Obama was elected 44th president of the United States in 2008."
            res = await ground_claims_async(claim, use_cache=False)
            self.assertEqual(res["corroboration_status"], "HISTORICAL_FACT_GROUNDED")
            self.assertGreaterEqual(res["corroboration_score"], 90.0)
            self.assertTrue(res["is_historical"])

        asyncio.run(run_test())

    @patch("News_Detection.grounding.fetch_news_rss")
    @patch("News_Detection.grounding.fetch_wikipedia_grounding")
    def test_fabricated_claim_with_historical_year_not_grounded(self, mock_wiki, mock_rss):
        """Milestone 8.2: Tests that a fabricated assertion with a past year is NOT falsely corroborated."""
        mock_rss.return_value = []
        # Wikipedia entry mentions Obama, but DOES NOT mention Martian Emperor or 1969
        mock_wiki.return_value = [
            {
                "title": "Barack Obama",
                "publisher": "Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Barack_Obama",
                "snippet": "Barack Obama was born in 1961 and served as US president from 2009 to 2017.",
                "published_at": "Encyclopedic Record",
                "is_authoritative": True,
                "type": "encyclopedia",
            }
        ]

        claim = "Barack Obama was crowned Martian Emperor in 1969."
        res = ground_claims(claim, use_cache=False)

        # Year 1969 does not align with Obama's summary or the absurd claim
        self.assertEqual(res["corroboration_status"], "UNVERIFIED_CLAIM")
        self.assertLessEqual(res["corroboration_score"], 10.0)
        self.assertFalse(res["is_historical"])


if __name__ == "__main__":
    unittest.main()
