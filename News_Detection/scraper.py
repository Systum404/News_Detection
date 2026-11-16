"""
Real-Time Article URL Scraper & Web Crawler Engine.

Extracts sanitized article text, headline, author, publication date,
and publisher metadata from any news article URL with SSRF protection,
paywall detection, and boilerplate removal.
"""

import ipaddress
import logging
import re
import socket
import urllib.parse
import urllib.request
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup

logger = logging.getLogger("news_detection.scraper")

# SSRF Protection: Private & Reserved IP Networks
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # IPv4 loopback
    ipaddress.ip_network("10.0.0.0/8"),       # RFC 1918 Private
    ipaddress.ip_network("172.16.0.0/12"),    # RFC 1918 Private
    ipaddress.ip_network("192.168.0.0/16"),   # RFC 1918 Private
    ipaddress.ip_network("169.254.0.0/16"),   # Link-Local
    ipaddress.ip_network("0.0.0.0/8"),        # Current network
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 Unique Local
    ipaddress.ip_network("fe80::/10"),        # IPv6 Link-Local
]

# Social media & video streaming platforms that lack standard article semantic bodies
BLOCKED_MEDIA_DOMAINS = {
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "tiktok.com",
    "facebook.com",
    "fb.com",
    "reddit.com",
    "threads.net",
}


def is_forbidden_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Checks if an IP address is private, loopback, link-local, reserved, multicast, or unspecified.
    Correctly unwraps IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1).
    """
    if getattr(ip_obj, "ipv4_mapped", None):
        ip_obj = ip_obj.ipv4_mapped

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    ):
        return True

    for net in BLOCKED_IP_NETWORKS:
        try:
            if ip_obj in net:
                return True
        except TypeError:
            continue
    return False

# Standard modern browser request headers
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8,es;q=0.7,*;q=0.5",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Elements that never contain genuine news content
BOILERPLATE_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "noscript",
    "iframe", "svg", "form", "button", "menu", "dialog"
]

BOILERPLATE_CLASSES_OR_IDS = [
    "cookie", "banner", "advertisement", "ad-container", "ad-slot", "sidebar",
    "social-share", "newsletter-signup", "related-posts", "comments-section",
    "menu-drawer", "popup", "paywall-prompt", "nav-container", "footer-links"
]


class ScraperError(Exception):
    """Base exception for URL scraping failures."""
    pass


class InvalidURLError(ScraperError):
    """Raised when the URL format or protocol is invalid."""
    pass


class SSRFBlockedError(ScraperError):
    """Raised when URL resolves to a forbidden private or local IP."""
    pass


class ContentExtractionError(ScraperError):
    """Raised when insufficient text can be extracted from the page."""
    pass


ALLOWED_SCRAPE_PORTS = {80, 443, 8080, 8443}


def validate_target_url(raw_url: str) -> urllib.parse.ParseResult:
    """
    Validates URL scheme, hostname, port, and applies SSRF security checks.
    Blocks localhost, private IP subnets, non-standard service ports, and userinfo credentials.
    """
    url = raw_url.strip()
    if not url:
        raise InvalidURLError("URL string cannot be empty.")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError(f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    if parsed.username or parsed.password:
        raise InvalidURLError("Embedded credentials (userinfo) in URLs are forbidden for security.")

    hostname = parsed.hostname
    if not hostname:
        raise InvalidURLError("Invalid URL: missing hostname.")

    host_lower = hostname.lower()
    for social in BLOCKED_MEDIA_DOMAINS:
        if host_lower == social or host_lower.endswith("." + social):
            raise InvalidURLError(
                f"Links from '{hostname}' cannot be scraped as standard news articles. "
                "Social media and streaming platforms require interactive authentication or dynamic rendering. "
                "Please copy and paste the text directly into the Claim / Text Input tab."
            )

    # 1. Block obvious local and loopback hostnames
    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise SSRFBlockedError(f"Access to local host '{hostname}' is blocked for security.")

    # 2. Check if hostname directly parses as an IP literal
    try:
        direct_ip = ipaddress.ip_address(hostname)
        if is_forbidden_ip(direct_ip):
            raise SSRFBlockedError(f"URL points to forbidden IP address '{hostname}'.")
    except ValueError:
        pass

    # 3. Port restrictions (prevent port scanning / internal protocol injection)
    if parsed.port and parsed.port not in ALLOWED_SCRAPE_PORTS:
        raise InvalidURLError(
            f"Port {parsed.port} is not permitted. Only standard web ports {sorted(ALLOWED_SCRAPE_PORTS)} are allowed."
        )

    # DNS Resolution Check for SSRF prevention
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            if is_forbidden_ip(ip_obj):
                raise SSRFBlockedError(
                    f"URL resolves to protected or private IP address '{ip_str}', which is forbidden."
                )
    except socket.gaierror:
        raise InvalidURLError(f"Unable to resolve domain '{hostname}'. Please check if the URL is correct.")

    return parsed


def clean_article_text(raw_text: str) -> str:
    """Normalizes whitespace and removes trailing/leading clutter."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw_text.splitlines()]
    paragraphs = []
    current_para = []
    for line in lines:
        if line:
            current_para.append(line)
        elif current_para:
            paragraphs.append(" ".join(current_para))
            current_para = []
    if current_para:
        paragraphs.append(" ".join(current_para))
    return "\n\n".join(paragraphs).strip()



def get_meta_content(soup: BeautifulSoup, **kwargs) -> Optional[str]:
    """Helper to safely extract content from meta tags."""
    tag = soup.find("meta", attrs=kwargs)
    if tag and getattr(tag, "attrs", None) and isinstance(tag.attrs, dict):
        content = tag.attrs.get("content")
        if content:
            return str(content).strip()
    return None


def extract_metadata(soup: BeautifulSoup, parsed_url: urllib.parse.ParseResult) -> Dict[str, Any]:
    """Extracts headline, author, publisher, date, and hero image from HTML metadata."""
    # 1. Headline / Title
    title = (
        get_meta_content(soup, property="og:title")
        or get_meta_content(soup, name="twitter:title")
    )

    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text().strip()

    # Clean site suffix from title (e.g. "News Title | BBC News" -> "News Title")
    if title:
        title = re.sub(r"\s+[-–—|]\s+[A-Za-z0-9\s\.\(\)]+$", "", title)

    # 2. Publisher
    publisher = get_meta_content(soup, property="og:site_name")

    if not publisher:
        # Fallback to domain name
        domain = parsed_url.hostname.replace("www.", "")
        parts = domain.split(".")
        if len(parts) >= 2 and len(parts[0]) <= 3:
            publisher = parts[1].capitalize()
        else:
            publisher = parts[0].capitalize()


    # 3. Author
    author = (
        get_meta_content(soup, name="author")
        or get_meta_content(soup, property="article:author")
    )

    # 4. Publication Date
    published_date = (
        get_meta_content(soup, property="article:published_time")
        or get_meta_content(soup, name="pubdate")
        or get_meta_content(soup, name="publish-date")
    )
    if not published_date:
        time_tag = soup.find("time")
        if time_tag and getattr(time_tag, "attrs", None) and isinstance(time_tag.attrs, dict):
            published_date = time_tag.attrs.get("datetime")

    # 5. Top Hero Image
    image_url = (
        get_meta_content(soup, property="og:image")
        or get_meta_content(soup, name="twitter:image")
    )
    if image_url and not image_url.startswith(("http://", "https://")):
        image_url = None

    return {
        "title": title or "Untitled News Article",
        "publisher": publisher or "Web Publication",
        "author": author,
        "published_date": published_date,
        "image_url": image_url,
    }


def extract_article_body(soup: BeautifulSoup) -> str:
    """
    Strips boilerplate and extracts genuine paragraph content from
    article semantic tags or main content containers.
    """
    # Remove boilerplate tags
    for tag_name in BOILERPLATE_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Remove elements with boilerplate class or id substrings
    for el in soup.find_all(True):
        attrs = getattr(el, "attrs", None)
        if not attrs or not isinstance(attrs, dict):
            continue
        if el.name in ("body", "html", "article", "main"):
            continue
        class_val = attrs.get("class", [])
        class_str = " ".join(class_val) if isinstance(class_val, list) else str(class_val)
        id_str = str(attrs.get("id", ""))
        combined = f"{class_str} {id_str}".lower()
        if any(b in combined for b in BOILERPLATE_CLASSES_OR_IDS):
            el.decompose()

    # Find the best container
    main_container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(class_=re.compile(r"article[-_]?body|story[-_]?content|post[-_]?content", re.I))
        or soup.body
    )


    if not main_container:
        return ""

    paragraphs: List[str] = []
    # Collect all paragraphs with substantive content
    for p in main_container.find_all("p"):
        text = p.get_text().strip()
        # Filter out one-word disclaimers or cookie snippets
        if len(text.split()) >= 4 and len(text) > 25:
            paragraphs.append(text)

    # If no paragraphs found in article tag, fallback to substantive div texts
    if not paragraphs:
        for div in main_container.find_all("div"):
            if len(div.find_all("p")) == 0:
                text = div.get_text().strip()
                if len(text.split()) >= 8 and len(text) > 40:
                    paragraphs.append(text)

    return clean_article_text("\n\n".join(paragraphs))


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    HTTP redirect handler that enforces strict SSRF validation on every redirect hop.
    Prevents attackers from using 301/302/303/307 redirects to bounce to internal networks,
    localhost, or cloud metadata endpoints.
    """
    def __init__(self, max_redirects: int = 5):
        super().__init__()
        self.max_redirects = max_redirects
        self._redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._redirect_count += 1
        if self._redirect_count > self.max_redirects:
            raise SSRFBlockedError(f"Exceeded maximum redirection limit of {self.max_redirects}.")
        # Strict validation of new redirect target
        validate_target_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_article_html(req: urllib.request.Request, timeout: float = 8.0) -> str:
    """
    Executes HTTP GET using a secure opener equipped with SafeRedirectHandler.
    Protects against SSRF redirect bypasses and limits memory consumption to 4MB.
    """
    opener = urllib.request.build_opener(SafeRedirectHandler())
    with opener.open(req, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise ContentExtractionError(
                f"URL returned non-HTML content type '{content_type}'. Cannot extract news article."
            )

        # Limit reading to 4MB to prevent memory exhaustion
        return response.read(4 * 1024 * 1024).decode("utf-8", errors="ignore")


def scrape_article(url: str, timeout: float = 8.0) -> Dict[str, Any]:
    """
    Fetches and extracts full article content and metadata from a URL.
    Enforces SSRF validation, size bounds, and content thresholds.
    """
    parsed_url = validate_target_url(url)
    target_url = parsed_url.geturl()

    req = urllib.request.Request(
        target_url,
        headers=BROWSER_HEADERS,
    )

    try:
        raw_html = fetch_article_html(req, timeout=timeout)
    except SSRFBlockedError:
        raise
    except urllib.error.HTTPError as http_err:
        if http_err.code in (401, 403):
            raise ContentExtractionError(
                f"Access restricted (HTTP {http_err.code}). The article may be behind a paywall or anti-scraping gate."
            )
        elif http_err.code == 404:
            raise ContentExtractionError("Article not found (HTTP 404). Please verify the link.")
        raise ContentExtractionError(f"HTTP error {http_err.code} fetching article.")
    except urllib.error.URLError as url_err:
        raise ContentExtractionError(f"Connection error reaching URL: {url_err.reason}")
    except socket.timeout:
        raise ContentExtractionError(f"Request timed out after {timeout} seconds.")
    except Exception as exc:
        raise ContentExtractionError(f"Error downloading article: {exc}")

    soup = BeautifulSoup(raw_html, "html.parser")
    metadata = extract_metadata(soup, parsed_url)
    article_text = extract_article_body(soup)

    # If body text is empty or too short, check for paywall barriers or combine headline
    if not article_text or len(article_text.split()) < 25:
        paywall_pattern = re.compile(
            r"paywall|subscription[-_]?required|subscriber[-_]?only|login[-_]?wall|gate[-_]?wall|sign[-_]?in to read|premium article|already a subscriber|subscribe now to",
            re.IGNORECASE,
        )
        if paywall_pattern.search(raw_html[:30000]):
            raise ContentExtractionError(
                "Paywalled or subscription-gated article detected. The complete news article body cannot be accessed publicly. "
                "Please copy and paste the subscriber text directly into the Claim / Text Input tab."
            )

        if metadata["title"] and len(metadata["title"].split()) >= 6:
            article_text = metadata["title"] + "\n\n" + (article_text or "")
        else:
            raise ContentExtractionError(
                "Unable to extract sufficient article text from this page. "
                "The content may require interactive JavaScript rendering, login authentication, or a paid subscription."
            )

    words = article_text.split()
    word_count = len(words)
    char_count = len(article_text)
    domain = parsed_url.hostname.replace("www.", "")

    return {
        "url": target_url,
        "domain": domain,
        "publisher": metadata["publisher"],
        "title": metadata["title"],
        "author": metadata["author"],
        "published_date": metadata["published_date"],
        "image_url": metadata["image_url"],
        "favicon": f"https://www.google.com/s2/favicons?domain={domain}&sz=64",
        "text": article_text,
        "snippet": article_text[:250] + ("..." if len(article_text) > 250 else ""),
        "word_count": word_count,
        "char_count": char_count,
    }
