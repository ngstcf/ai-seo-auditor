import os
import asyncio
import json
import re
import csv
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Crawl4AI Imports
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

# LLM Client
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# --- 1. Helper Functions ---
def extract_schema_tags(html: str) -> Dict[str, Any]:
    """
    Extract JSON-LD schema blocks from HTML.
    Returns both raw schemas and parsed data for analysis.
    """
    empty = {"found": False, "schemas": [], "types": [], "date_published": None,
             "date_modified": None, "authors": [], "has_faq_schema": False}
    if not html:
        return empty

    pattern = r'<script\s+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)

    if not matches:
        return empty

    schemas = []
    schema_types = set()
    date_published = None
    date_modified = None
    authors = []

    def _extract_from_obj(obj):
        nonlocal date_published, date_modified
        if not isinstance(obj, dict):
            return
        t = obj.get('@type', '')
        type_str = t if isinstance(t, str) else str(t)
        schema_types.add(type_str)

        if not date_published and obj.get('datePublished'):
            date_published = str(obj['datePublished'])
        if not date_modified and obj.get('dateModified'):
            date_modified = str(obj['dateModified'])

        author = obj.get('author')
        if author:
            if isinstance(author, dict):
                name = author.get('name')
                if name and name not in authors:
                    authors.append(name)
            elif isinstance(author, list):
                for a in author:
                    if isinstance(a, dict):
                        name = a.get('name')
                        if name and name not in authors:
                            authors.append(name)

    for content in matches:
        try:
            clean_content = re.sub(r'\s+', ' ', content).strip()
            parsed = json.loads(clean_content)

            if isinstance(parsed, dict):
                _extract_from_obj(parsed)
                for item in parsed.get('@graph', []):
                    _extract_from_obj(item)
            elif isinstance(parsed, list):
                for item in parsed:
                    _extract_from_obj(item)

            schemas.append(clean_content)
        except json.JSONDecodeError:
            schemas.append(clean_content)

    has_faq_schema = any(t.lower() in ('faqpage', 'faq') for t in schema_types)

    return {
        "found": True,
        "schemas": schemas,
        "types": list(schema_types),
        "date_published": date_published,
        "date_modified": date_modified,
        "authors": authors,
        "has_faq_schema": has_faq_schema,
    }

def check_llms_txt(base_url: str) -> Optional[str]:
    """
    Check if llms.txt exists at the root domain.
    """
    import requests
    
    try:
        llms_url = f"{base_url.rstrip('/')}/llms.txt"
        response = requests.get(llms_url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; SEO-Auditor/1.0)'
        })
        
        if response.status_code == 200:
            return "Present"
        else:
            return "Not Found"
    except Exception:
        return "Not Found"

def check_robots_txt(base_url: str) -> Dict[str, Any]:
    import requests

    AI_BOTS = [
        "GPTBot", "ChatGPT-User", "OAI-SearchBot", "ClaudeBot",
        "anthropic-ai", "PerplexityBot", "GoogleOther", "Google-Extended",
        "Amazonbot", "Applebot-Extended", "Bytespider", "CCBot",
    ]

    try:
        url = f"{base_url.rstrip('/')}/robots.txt"
        resp = requests.get(url, timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; SEO-Auditor/1.0)'
        })
        if resp.status_code != 200:
            return {"found": False, "ai_bot_status": {}, "blocked_count": 0,
                    "wildcard_disallow_all": False, "summary": "robots.txt not found"}

        content_type = resp.headers.get("Content-Type", "")
        text = resp.text
        if "text/" not in content_type and "User-agent" not in text[:500]:
            return {"found": False, "ai_bot_status": {}, "blocked_count": 0,
                    "wildcard_disallow_all": False, "summary": "robots.txt not found (invalid content)"}

        sections: Dict[str, List[str]] = {}
        current_agents: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("user-agent:"):
                agent = line.split(":", 1)[1].strip()
                current_agents = [agent]
                sections.setdefault(agent, [])
            elif current_agents:
                for agent in current_agents:
                    sections.setdefault(agent, []).append(line)

        wildcard_disallow_all = False
        wildcard_directives = sections.get("*", [])
        for d in wildcard_directives:
            if d.lower().startswith("disallow:") and d.split(":", 1)[1].strip() == "/":
                wildcard_disallow_all = True
                break

        ai_bot_status = {}
        blocked_count = 0
        for bot in AI_BOTS:
            if bot in sections:
                directives = sections[bot]
                disallows = [d for d in directives if d.lower().startswith("disallow:")]
                allows = [d for d in directives if d.lower().startswith("allow:")]
                has_root_block = any(d.split(":", 1)[1].strip() == "/" for d in disallows)
                has_root_allow = any(d.split(":", 1)[1].strip() == "/" for d in allows)
                if has_root_block and not has_root_allow:
                    ai_bot_status[bot] = "Blocked"
                    blocked_count += 1
                elif disallows and not has_root_block:
                    ai_bot_status[bot] = "Partially Blocked"
                else:
                    ai_bot_status[bot] = "Allowed"
            elif wildcard_disallow_all:
                ai_bot_status[bot] = "Blocked (wildcard)"
                blocked_count += 1
            else:
                ai_bot_status[bot] = "Not Mentioned"

        summary = f"{blocked_count} of {len(AI_BOTS)} AI crawlers blocked"
        if wildcard_disallow_all:
            summary += " (wildcard Disallow: / active)"

        return {
            "found": True,
            "ai_bot_status": ai_bot_status,
            "blocked_count": blocked_count,
            "wildcard_disallow_all": wildcard_disallow_all,
            "summary": summary,
        }
    except Exception:
        return {"found": False, "ai_bot_status": {}, "blocked_count": 0,
                "wildcard_disallow_all": False, "summary": "robots.txt check failed"}


def check_sitemap_xml(base_url: str) -> Dict[str, Any]:
    import requests

    try:
        url = f"{base_url.rstrip('/')}/sitemap.xml"
        resp = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; SEO-Auditor/1.0)'
        })
        if resp.status_code != 200:
            return {"found": False, "url_count": 0, "has_lastmod": False,
                    "most_recent_lastmod": None, "lastmod_coverage": 0.0,
                    "is_index": False, "summary": "sitemap.xml not found"}

        text = resp.text
        is_index = "<sitemapindex" in text.lower()

        locs = re.findall(r'<loc>(.*?)</loc>', text, re.IGNORECASE)
        lastmods = re.findall(r'<lastmod>(.*?)</lastmod>', text, re.IGNORECASE)
        url_count = len(locs)
        lastmod_count = len(lastmods)
        coverage = (lastmod_count / url_count * 100) if url_count > 0 else 0.0

        most_recent = None
        if lastmods:
            sorted_dates = sorted(lastmods, reverse=True)
            most_recent = sorted_dates[0]

        label = "sitemap index" if is_index else "sitemap"
        summary = f"{label}: {url_count} URLs, {coverage:.0f}% with lastmod"
        if most_recent:
            summary += f", newest: {most_recent[:10]}"

        return {
            "found": True,
            "url_count": url_count,
            "has_lastmod": lastmod_count > 0,
            "most_recent_lastmod": most_recent,
            "lastmod_coverage": coverage,
            "is_index": is_index,
            "summary": summary,
        }
    except Exception:
        return {"found": False, "url_count": 0, "has_lastmod": False,
                "most_recent_lastmod": None, "lastmod_coverage": 0.0,
                "is_index": False, "summary": "sitemap.xml check failed"}


def check_ai_discovery_files(base_url: str) -> Dict[str, Any]:
    import requests

    files_to_check = {
        "llms_full_txt": "/llms-full.txt",
        "ai_json": "/.well-known/ai.json",
    }
    results = {}
    for key, path in files_to_check.items():
        try:
            url = f"{base_url.rstrip('/')}{path}"
            resp = requests.head(url, timeout=5, allow_redirects=True, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; SEO-Auditor/1.0)'
            })
            results[key] = "Present" if resp.status_code == 200 else "Not Found"
        except Exception:
            results[key] = "Not Found"

    present = [k for k, v in results.items() if v == "Present"]
    summary = f"{len(present)} additional AI discovery files found" if present else "No additional AI discovery files"
    results["summary"] = summary
    return results


def extract_structured_content(html: str) -> Dict[str, Any]:
    """
    Analyze content structure for GEO/AEO optimization.
    """
    # Check for FAQ sections
    faq_pattern = r'<(div|section)[^>]*?(faq|question|answer)[^>]*?>'
    has_faq = bool(re.search(faq_pattern, html, re.IGNORECASE))
    
    # Check for lists
    list_count = len(re.findall(r'<(ul|ol)', html, re.IGNORECASE))
    
    # Check for tables
    table_count = len(re.findall(r'<table', html, re.IGNORECASE))
    
    # Check for headings structure
    h2_count = len(re.findall(r'<h2', html, re.IGNORECASE))
    h3_count = len(re.findall(r'<h3', html, re.IGNORECASE))
    
    definition_list_count = len(re.findall(r'<dl[\s>]', html, re.IGNORECASE))
    has_accordion = bool(re.search(r'<details[\s>]', html, re.IGNORECASE))

    return {
        "has_faq": has_faq,
        "list_count": list_count,
        "table_count": table_count,
        "heading_count": h2_count + h3_count,
        "definition_list_count": definition_list_count,
        "has_accordion": has_accordion,
        "has_structured_content": list_count > 0 or table_count > 0 or has_faq or definition_list_count > 0
    }

def html_to_markdown_simple(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
        tag.decompose()
    lines = []
    for el in soup.find_all(['h1','h2','h3','h4','h5','h6','p','li','td','th','blockquote','pre']):
        text = el.get_text(' ', strip=True)
        if not text:
            continue
        if el.name.startswith('h'):
            level = int(el.name[1])
            lines.append(f"\n{'#' * level} {text}\n")
        elif el.name == 'li':
            lines.append(f"- {text}")
        elif el.name == 'blockquote':
            lines.append(f"> {text}")
        else:
            lines.append(text)
    return '\n\n'.join(lines)


def extract_links_from_html(html: str, page_url: str) -> Dict[str, List[Dict]]:
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, urljoin
    soup = BeautifulSoup(html, 'html.parser')
    parsed_page = urlparse(page_url)
    page_domain = parsed_page.netloc
    internal, external = [], []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(page_url, href)
        text = a.get_text(strip=True)
        parsed_link = urlparse(full_url)
        entry = {"href": full_url, "text": text, "base_domain": parsed_link.netloc}
        if parsed_link.netloc == page_domain:
            internal.append(entry)
        elif parsed_link.scheme in ('http', 'https'):
            external.append(entry)
    return {"internal": internal, "external": external}


def extract_meta_tags(html: str, metadata: Optional[dict] = None) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser') if html else None

    def _meta(prop=None, name=None):
        if metadata:
            if prop and prop in metadata:
                return metadata[prop]
            if name and name in metadata:
                return metadata[name]
        if not soup:
            return None
        if prop:
            tag = soup.find('meta', attrs={'property': prop}) or soup.find('meta', attrs={'name': prop})
            if tag:
                return tag.get('content')
        if name:
            tag = soup.find('meta', attrs={'name': name}) or soup.find('meta', attrs={'property': name})
            if tag:
                return tag.get('content')
        return None

    og_title = _meta(prop='og:title')
    og_description = _meta(prop='og:description')
    og_type = _meta(prop='og:type')
    og_image = _meta(prop='og:image')

    canonical = None
    if soup:
        link = soup.find('link', attrs={'rel': 'canonical'})
        if link:
            canonical = link.get('href')

    meta_description = _meta(name='description')
    meta_author = _meta(name='author')

    core_og = [og_title, og_description, og_type, og_image]
    og_completeness = sum(1 for v in core_og if v)

    summary_parts = []
    if og_completeness == 4:
        summary_parts.append("OG tags complete")
    elif og_completeness > 0:
        summary_parts.append(f"OG tags {og_completeness}/4")
    else:
        summary_parts.append("No OG tags")
    if not canonical:
        summary_parts.append("no canonical")
    if not meta_description:
        summary_parts.append("no meta description")

    return {
        "og_title": og_title,
        "og_description": og_description,
        "og_type": og_type,
        "og_image": og_image,
        "canonical": canonical,
        "meta_description": meta_description,
        "meta_author": meta_author,
        "has_og_tags": og_completeness > 0,
        "og_completeness": og_completeness,
        "summary": "; ".join(summary_parts),
    }


def extract_content_freshness(html: str, metadata: Optional[dict] = None,
                               schema_data: Optional[dict] = None) -> Dict[str, Any]:
    from dateutil import parser as dateutil_parser
    from datetime import datetime, timezone

    date_published = None
    date_modified = None
    sources: List[str] = []

    def _try_parse(val):
        if not val:
            return None
        try:
            return dateutil_parser.parse(str(val))
        except (ValueError, OverflowError):
            return None

    if schema_data:
        dp = _try_parse(schema_data.get('date_published'))
        if dp:
            date_published = dp
            sources.append("schema")
        dm = _try_parse(schema_data.get('date_modified'))
        if dm:
            date_modified = dm
            if "schema" not in sources:
                sources.append("schema")

    if metadata:
        if not date_published:
            dp = _try_parse(metadata.get('article:published_time'))
            if dp:
                date_published = dp
                sources.append("meta")
        if not date_modified:
            dm = _try_parse(metadata.get('article:modified_time'))
            if dm:
                date_modified = dm
                if "meta" not in sources:
                    sources.append("meta")

    if html and (not date_published or not date_modified):
        time_tags = re.findall(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, re.IGNORECASE)
        for t in time_tags:
            parsed = _try_parse(t)
            if parsed:
                if not date_published:
                    date_published = parsed
                    sources.append("html")
                elif not date_modified and parsed != date_published:
                    date_modified = parsed
                    if "html" not in sources:
                        sources.append("html")
                    break

    has_visible_dates = bool(html and re.search(
        r'<time[\s>]|class=["\'][^"\']*\b(date|published|updated|modified)\b', html, re.IGNORECASE
    ))

    reference_date = date_modified or date_published
    days_since = None
    freshness_status = "Unknown"
    if reference_date:
        if reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_since = (now - reference_date).days
        if days_since < 90:
            freshness_status = "Fresh"
        elif days_since < 365:
            freshness_status = "Aging"
        elif days_since < 730:
            freshness_status = "Stale"
        else:
            freshness_status = "Very Stale"

    dp_str = date_published.isoformat() if date_published else None
    dm_str = date_modified.isoformat() if date_modified else None
    summary = f"{freshness_status}"
    if days_since is not None:
        summary += f" ({days_since} days)"
    if not has_visible_dates:
        summary += ", no visible dates on page"

    return {
        "date_published": dp_str,
        "date_modified": dm_str,
        "days_since_modified": days_since,
        "freshness_status": freshness_status,
        "has_visible_dates": has_visible_dates,
        "date_sources": sources,
        "summary": summary,
    }


def analyze_heading_hierarchy(html: str) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser') if html else None

    if not soup:
        return {"h1_count": 0, "h2_count": 0, "h3_count": 0, "total_headings": 0,
                "multiple_h1": False, "hierarchy_valid": True, "skipped_levels": [],
                "question_headings_count": 0, "question_headings_pct": 0.0,
                "heading_texts": [], "summary": "No HTML to analyze"}

    headings = []
    for tag in soup.find_all(re.compile(r'^h[1-6]$', re.IGNORECASE)):
        level = int(tag.name[1])
        text = tag.get_text(strip=True)
        if text:
            headings.append((level, text))

    counts = {i: 0 for i in range(1, 7)}
    for level, _ in headings:
        counts[level] += 1

    skipped_levels = []
    hierarchy_valid = True
    for i in range(1, len(headings)):
        prev_level = headings[i - 1][0]
        curr_level = headings[i][0]
        if curr_level > prev_level + 1:
            skipped_levels.append(f"H{prev_level}->H{curr_level}")
            hierarchy_valid = False

    question_words = {"how", "what", "why", "when", "which", "who", "is", "are", "can", "do", "does", "should", "will", "where"}
    question_count = 0
    for _, text in headings:
        text_lower = text.lower().strip()
        if "?" in text or text_lower.split()[0] in question_words if text_lower.split() else False:
            question_count += 1

    total = len(headings)
    q_pct = (question_count / total * 100) if total > 0 else 0.0

    parts = []
    if counts[1] > 1:
        parts.append(f"multiple H1s ({counts[1]})")
    if not hierarchy_valid:
        parts.append(f"skipped levels: {', '.join(skipped_levels[:3])}")
    if question_count > 0:
        parts.append(f"{question_count} question headings")
    summary = "; ".join(parts) if parts else f"{total} headings, hierarchy OK"

    return {
        "h1_count": counts[1],
        "h2_count": counts[2],
        "h3_count": counts[3],
        "total_headings": total,
        "multiple_h1": counts[1] > 1,
        "hierarchy_valid": hierarchy_valid,
        "skipped_levels": skipped_levels,
        "question_headings_count": question_count,
        "question_headings_pct": q_pct,
        "heading_texts": [t for _, t in headings[:10]],
        "summary": summary,
    }


def analyze_links(links: Optional[dict], page_url: str) -> Dict[str, Any]:
    if not links:
        return {"internal_count": 0, "external_count": 0, "authoritative_external": 0,
                "authoritative_domains": [], "empty_anchor_count": 0,
                "has_citations": False, "summary": "No link data available"}

    internal = links.get("internal", [])
    external = links.get("external", [])
    internal_count = len(internal)
    external_count = len(external)

    authority_tlds = {".gov", ".edu"}
    authority_domains = set()
    generic_anchors = {"click here", "here", "read more", "learn more", "link", "this", "more"}
    empty_anchor_count = 0

    for link in external:
        href = link.get("href", "") if isinstance(link, dict) else str(link)
        text = (link.get("text", "") if isinstance(link, dict) else "").strip().lower()
        domain = link.get("base_domain", href) if isinstance(link, dict) else href

        if any(tld in href.lower() for tld in authority_tlds):
            authority_domains.add(domain if domain else href)

        if not text or text in generic_anchors:
            empty_anchor_count += 1

    for link in internal:
        text = (link.get("text", "") if isinstance(link, dict) else "").strip().lower()
        if not text or text in generic_anchors:
            empty_anchor_count += 1

    auth_count = len(authority_domains)
    parts = [f"{internal_count} internal, {external_count} external"]
    if auth_count > 0:
        parts.append(f"{auth_count} authoritative")
    if empty_anchor_count > 3:
        parts.append(f"{empty_anchor_count} weak anchors")

    return {
        "internal_count": internal_count,
        "external_count": external_count,
        "authoritative_external": auth_count,
        "authoritative_domains": list(authority_domains)[:5],
        "empty_anchor_count": empty_anchor_count,
        "has_citations": external_count > 0,
        "summary": "; ".join(parts),
    }


def extract_eeat_signals(html: str, metadata: Optional[dict] = None,
                          schema_data: Optional[dict] = None) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser') if html else None

    has_author_byline = False
    author_name = None
    has_author_bio = False
    has_person_schema = False
    has_credentials = False
    credential_indicators: List[str] = []
    has_fact_check = False

    if schema_data:
        if schema_data.get('authors'):
            author_name = schema_data['authors'][0]
            has_author_byline = True
        types_lower = [t.lower() for t in schema_data.get('types', [])]
        has_person_schema = 'person' in types_lower

    if metadata and not author_name:
        meta_author = metadata.get('author')
        if meta_author:
            author_name = meta_author
            has_author_byline = True

    if soup:
        byline_selectors = [
            {'attrs': {'rel': 'author'}},
            {'attrs': {'class': re.compile(r'author|byline', re.IGNORECASE)}},
            {'attrs': {'itemprop': 'author'}},
        ]
        for sel in byline_selectors:
            tag = soup.find(**sel)
            if tag:
                has_author_byline = True
                if not author_name:
                    author_name = tag.get_text(strip=True)[:100]
                break

        bio_selectors = [
            {'attrs': {'class': re.compile(r'author.?bio|about.?author|writer.?bio', re.IGNORECASE)}},
            {'attrs': {'id': re.compile(r'author.?bio|about.?author', re.IGNORECASE)}},
        ]
        for sel in bio_selectors:
            if soup.find(**sel):
                has_author_bio = True
                break

        fact_check_pattern = re.compile(
            r'(reviewed\s+by|fact[- ]?checked?\s+by|medically\s+reviewed|verified\s+by)',
            re.IGNORECASE
        )
        if soup.find(string=fact_check_pattern):
            has_fact_check = True

    if html:
        cred_patterns = [
            (r'\bPh\.?D\.?\b', "PhD"),
            (r'\bM\.?D\.?\b', "MD"),
            (r'\bDr\.\s', "Dr."),
            (r'\bProfessor\b', "Professor"),
            (r'\bcertified\b', "Certified"),
            (r'\b\d+\+?\s*years?\s*(of\s+)?experience\b', "Years of experience"),
        ]
        for pattern, label in cred_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                has_credentials = True
                if label not in credential_indicators:
                    credential_indicators.append(label)

    signals = [has_author_byline, has_author_bio, has_person_schema,
               has_credentials, has_fact_check, bool(author_name)]
    signal_count = sum(signals)

    parts = []
    if author_name:
        parts.append(f"author: {author_name[:40]}")
    if has_author_bio:
        parts.append("bio present")
    if has_credentials:
        parts.append(f"credentials: {', '.join(credential_indicators[:3])}")
    if has_fact_check:
        parts.append("fact-checked")
    summary = "; ".join(parts) if parts else "No E-E-A-T signals detected"

    return {
        "has_author_byline": has_author_byline,
        "author_name": author_name,
        "has_author_bio": has_author_bio,
        "has_person_schema": has_person_schema,
        "has_credentials": has_credentials,
        "credential_indicators": credential_indicators,
        "has_fact_check": has_fact_check,
        "eeat_signal_count": signal_count,
        "summary": summary,
    }


def analyze_content_quality(markdown: str) -> Dict[str, Any]:
    if not markdown:
        return {"word_count": 0, "paragraph_count": 0, "avg_paragraph_length": 0,
                "short_paragraphs_pct": 0.0, "statistic_count": 0, "definition_count": 0,
                "quotable_sentences": 0, "has_sufficient_depth": False,
                "summary": "No content to analyze"}

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', markdown) if p.strip()]
    words = markdown.split()
    word_count = len(words)
    para_count = len(paragraphs)
    para_lengths = [len(p.split()) for p in paragraphs]
    avg_length = sum(para_lengths) / para_count if para_count > 0 else 0
    short_pct = (sum(1 for l in para_lengths if l <= 75) / para_count * 100) if para_count > 0 else 0.0

    stat_patterns = [
        r'\d+\.?\d*\s*%',
        r'\$\d+',
        r'\d+\.?\d*\s*(million|billion|thousand|trillion)',
        r'according\s+to\b',
        r'\[\d+\]',
        r'\(\d{4}\)',
    ]
    stat_count = 0
    for pat in stat_patterns:
        stat_count += len(re.findall(pat, markdown, re.IGNORECASE))

    def_patterns = [
        r'\b\w+\s+(?:is|are)\s+defined\s+as\b',
        r'\b\w+\s+refers?\s+to\b',
        r'\b\w+\s+(?:is|are)\s+(?:a|an|the)\s+\w+',
    ]
    def_count = 0
    for pat in def_patterns:
        def_count += len(re.findall(pat, markdown, re.IGNORECASE))

    sentences = re.split(r'[.!?]+', markdown)
    quotable = 0
    for s in sentences:
        s = s.strip()
        s_words = s.split()
        if 5 <= len(s_words) <= 30:
            if re.search(r'\d+\.?\d*\s*%|\$\d+|\d+\.?\d*x\b', s):
                quotable += 1

    has_depth = word_count > 300

    parts = [f"{word_count} words, {para_count} paragraphs"]
    if stat_count > 0:
        parts.append(f"{stat_count} statistics")
    if quotable > 0:
        parts.append(f"{quotable} quotable statements")
    if not has_depth:
        parts.append("thin content (<300 words)")

    return {
        "word_count": word_count,
        "paragraph_count": para_count,
        "avg_paragraph_length": round(avg_length),
        "short_paragraphs_pct": round(short_pct, 1),
        "statistic_count": stat_count,
        "definition_count": def_count,
        "quotable_sentences": quotable,
        "has_sufficient_depth": has_depth,
        "summary": "; ".join(parts),
    }


# --- 2. LLM Analysis Function ---
async def analyze_content_with_llm(
    content: str,
    schema_data: Dict[str, Any],
    structured_content: Dict[str, Any],
    llms_txt_status: str,
    url: str,
    robots_data: Dict[str, Any],
    meta_tags: Dict[str, Any],
    freshness: Dict[str, Any],
    headings: Dict[str, Any],
    link_data: Dict[str, Any],
    eeat_signals: Dict[str, Any],
    content_quality: Dict[str, Any],
    sitemap_data: Dict[str, Any],
    ai_files_data: Dict[str, Any],
):
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        print("❌ Error: LLM_API_KEY missing in .env")
        return None

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    system_prompt = """You are an expert AI SEO Auditor specializing in GEO (Generative Engine Optimization) and AEO (Answer Engine Optimization) for 2025-2026.

Analyze content for visibility in AI search engines (ChatGPT, Perplexity, Claude, Google AI Overviews, Gemini, Copilot).

CRITICAL FACTORS TO EVALUATE:

1. ROBOTS.TXT AI CRAWLER ACCESS (Critical)
   - Are major AI crawlers (GPTBot, ClaudeBot, PerplexityBot, etc.) blocked?
   - 40%+ of sites accidentally block AI crawlers via wildcard rules
   - Sites blocking AI crawlers cannot be indexed by AI search engines

2. SCHEMA MARKUP (High Priority)
   - Presence of JSON-LD structured data
   - Critical types: Article, FAQPage, Organization, LocalBusiness, Product, Person, HowTo, Review/AggregateRating
   - Schema completeness: datePublished, dateModified, author fields
   - Schema is now core infrastructure for AI understanding (confirmed by Google & Microsoft 2025)

3. LLMS.TXT & AI DISCOVERY (New Standard)
   - /llms.txt presence (curated index for AI)
   - /llms-full.txt (complete content file — gets 3-4x more AI visits)
   - /.well-known/ai.json discovery manifest

4. OPENGRAPH & META TAGS (High Priority)
   - og:title, og:description, og:type, og:image present and well-formed
   - Canonical URL set correctly
   - Meta description present and concise (150-160 chars)
   - Meta author tag

5. CONTENT FRESHNESS (High Priority)
   - datePublished and dateModified in schema and/or meta tags
   - 85% of AI-cited content is from the last 2 years
   - Pages not updated within 90 days are 3x more likely to lose AI citations
   - Visible dates on page increase trust signals

6. CONTENT STRUCTURE FOR AI (Essential)
   - Clear Q&A format sections
   - Lists, tables, definition lists (AI-friendly formats)
   - Concise paragraphs (2-4 sentences optimal for AI extraction)
   - Question-style headings increase citation likelihood by 40%
   - Direct answers in first 40-60 words of each section
   - Accordion/details elements for FAQ content

7. HEADING HIERARCHY (Important)
   - Single H1 per page
   - Proper H1->H2->H3 nesting without skipping levels
   - Question-format headings for AI query matching

8. LINK PROFILE (Important)
   - Internal linking for topical authority
   - External citations to authoritative sources (.gov, .edu, research)
   - Descriptive anchor text (not "click here")

9. E-E-A-T SIGNALS (Trust Factor)
   - Author bylines, bios, and Person schema
   - Credential indicators (PhD, MD, certifications)
   - Fact-check / review attribution
   - Expert quotes and citations
   - Authority signals (consistent brand entity)

10. CONTENT QUALITY METRICS
    - Sufficient depth (300+ words)
    - Statistical/data density (stat every 150-200 words)
    - Quotable definitions and statements
    - First-party research and original data

11. SEMANTIC CLARITY
    - Topic focus and clarity
    - Context-rich language and entity mentions
    - Consistent terminology

OUTPUT ONLY VALID JSON (no markdown):
{
  "title": "Page Title",
  "url": "URL",
  "overall_score": 0-100,
  "ai_readiness": "Poor/Fair/Good/Excellent",
  "metrics": [
    {"category": "Robots.txt AI Access", "score": 0-100, "status": "Pass/Warning/Fail", "details": "...", "blocked_bots": []},
    {"category": "Schema Markup", "score": 0-100, "status": "Pass/Warning/Fail", "details": "...", "types_found": []},
    {"category": "AI Discovery Files", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "OpenGraph & Meta", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "Content Freshness", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "Content Structure", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "Heading Structure", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "Link Profile", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "E-E-A-T Signals", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "Content Quality", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."},
    {"category": "Semantic Clarity", "score": 0-100, "status": "Pass/Warning/Fail", "details": "..."}
  ],
  "recommendations": [
    {"priority": "Critical/High/Medium/Low", "category": "...", "action": "Specific actionable recommendation", "impact": "Expected improvement"}
  ],
  "strengths": ["List of what's working well"],
  "ai_citation_potential": "Low/Medium/High"
}"""

    user_prompt = f"""Analyze this page for AI search optimization (GEO/AEO):

URL: {url}

=== ROBOTS.TXT ANALYSIS (Domain-Level) ===
Found: {robots_data['found']}
AI Bots Blocked: {robots_data['blocked_count']}/12
Wildcard Disallow All: {robots_data['wildcard_disallow_all']}
Bot Status: {json.dumps(robots_data.get('ai_bot_status', {}), indent=None)}

=== LLMS.TXT & AI DISCOVERY ===
llms.txt: {llms_txt_status}
llms-full.txt: {ai_files_data.get('llms_full_txt', 'Not Checked')}
.well-known/ai.json: {ai_files_data.get('ai_json', 'Not Checked')}

=== SITEMAP ===
Found: {sitemap_data['found']}
{sitemap_data['summary']}

=== SCHEMA.ORG DATA ===
Found: {schema_data['found']}
Types: {', '.join(schema_data['types']) if schema_data['types'] else 'None'}
Has FAQ Schema: {schema_data.get('has_faq_schema', False)}
Schema Authors: {', '.join(schema_data.get('authors', [])) or 'None'}
Schema datePublished: {schema_data.get('date_published', 'Not set')}
Schema dateModified: {schema_data.get('date_modified', 'Not set')}
Raw Data: {str(schema_data['schemas'][:2000]) if schema_data['schemas'] else 'No schema found'}

=== OPENGRAPH & META TAGS ===
OG Title: {meta_tags.get('og_title', 'Missing')}
OG Description: {str(meta_tags.get('og_description', 'Missing'))[:200]}
OG Type: {meta_tags.get('og_type', 'Missing')}
OG Image: {'Present' if meta_tags.get('og_image') else 'Missing'}
Canonical: {meta_tags.get('canonical', 'Missing')}
Meta Description: {str(meta_tags.get('meta_description', 'Missing'))[:200]}
Meta Author: {meta_tags.get('meta_author', 'Missing')}
OG Completeness: {meta_tags.get('og_completeness', 0)}/4

=== CONTENT FRESHNESS ===
Published: {freshness.get('date_published', 'Unknown')}
Last Modified: {freshness.get('date_modified', 'Unknown')}
Days Since Modified: {freshness.get('days_since_modified', 'Unknown')}
Freshness Status: {freshness.get('freshness_status', 'Unknown')}
Visible Dates on Page: {freshness.get('has_visible_dates', False)}

=== CONTENT STRUCTURE ===
Has FAQ: {structured_content['has_faq']}
Lists: {structured_content['list_count']}
Tables: {structured_content['table_count']}
Definition Lists: {structured_content.get('definition_list_count', 0)}
Accordions: {structured_content.get('has_accordion', False)}
Headings: {structured_content['heading_count']}

=== HEADING HIERARCHY ===
H1 Count: {headings.get('h1_count', 0)}
Multiple H1: {headings.get('multiple_h1', False)}
Hierarchy Valid: {headings.get('hierarchy_valid', True)}
Skipped Levels: {headings.get('skipped_levels', [])}
Question Headings: {headings.get('question_headings_count', 0)}/{headings.get('total_headings', 0)}
Sample Headings: {headings.get('heading_texts', [])[:8]}

=== LINK PROFILE ===
Internal Links: {link_data.get('internal_count', 0)}
External Links: {link_data.get('external_count', 0)}
Authoritative External (.gov/.edu): {link_data.get('authoritative_external', 0)}
Authoritative Domains: {link_data.get('authoritative_domains', [])}
Weak Anchor Text Count: {link_data.get('empty_anchor_count', 0)}

=== E-E-A-T SIGNALS (Programmatic) ===
Author Byline: {eeat_signals.get('has_author_byline', False)}
Author Name: {eeat_signals.get('author_name', 'Not found')}
Author Bio Section: {eeat_signals.get('has_author_bio', False)}
Person Schema: {eeat_signals.get('has_person_schema', False)}
Credentials Found: {eeat_signals.get('credential_indicators', [])}
Fact-Check Attribution: {eeat_signals.get('has_fact_check', False)}
E-E-A-T Signal Count: {eeat_signals.get('eeat_signal_count', 0)}/6

=== CONTENT QUALITY ===
Word Count: {content_quality.get('word_count', 0)}
Paragraphs: {content_quality.get('paragraph_count', 0)}
Avg Paragraph Length: {content_quality.get('avg_paragraph_length', 0)} words
Short Paragraphs: {content_quality.get('short_paragraphs_pct', 0)}%
Statistics Found: {content_quality.get('statistic_count', 0)}
Definitions Found: {content_quality.get('definition_count', 0)}
Quotable Sentences: {content_quality.get('quotable_sentences', 0)}

=== PAGE CONTENT (First 10000 chars) ===
{content[:10000]}

Provide comprehensive analysis with all 11 metric categories scored."""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ LLM Call Failed: {e}")
        return None

# --- 3. Enhanced CSV Output ---
def save_to_csv(reports: List[Dict], filename="ai_seo_report_2025.csv"):
    """Save detailed reports with 2025 metrics"""
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Date", "URL", "Title", "Overall Score", "AI Readiness",
                "Robots.txt Score", "AI Bots Blocked",
                "Schema Score", "Schema Types",
                "AI Discovery Score",
                "OpenGraph Score",
                "Content Freshness Score", "Days Since Modified",
                "Content Structure Score",
                "Heading Structure Score",
                "Link Profile Score", "Internal Links", "External Links",
                "E-E-A-T Score",
                "Content Quality Score", "Word Count",
                "Semantic Score",
                "AI Citation Potential", "Top Priority Action", "Strengths"
            ])

        for r in reports:
            metrics = {m['category']: m for m in r.get('metrics', [])}
            rec = r.get('recommendations', [])
            top_rec = next((x['action'] for x in rec if x.get('priority') == 'Critical'),
                          rec[0]['action'] if rec else "None")

            schema_metric = metrics.get('Schema Markup', {})
            schema_types = ', '.join(schema_metric.get('types_found', []))

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M"),
                r.get('url'),
                r.get('title'),
                r.get('overall_score'),
                r.get('ai_readiness'),
                metrics.get('Robots.txt AI Access', {}).get('score', 'N/A'),
                ', '.join(metrics.get('Robots.txt AI Access', {}).get('blocked_bots', [])) or 'None',
                metrics.get('Schema Markup', {}).get('score', 'N/A'),
                schema_types or 'None',
                metrics.get('AI Discovery Files', {}).get('score', 'N/A'),
                metrics.get('OpenGraph & Meta', {}).get('score', 'N/A'),
                metrics.get('Content Freshness', {}).get('score', 'N/A'),
                metrics.get('Content Freshness', {}).get('details', 'N/A'),
                metrics.get('Content Structure', {}).get('score', 'N/A'),
                metrics.get('Heading Structure', {}).get('score', 'N/A'),
                metrics.get('Link Profile', {}).get('score', 'N/A'),
                metrics.get('Link Profile', {}).get('details', 'N/A'),
                metrics.get('Link Profile', {}).get('details', 'N/A'),
                metrics.get('E-E-A-T Signals', {}).get('score', 'N/A'),
                metrics.get('Content Quality', {}).get('score', 'N/A'),
                metrics.get('Content Quality', {}).get('details', 'N/A'),
                metrics.get('Semantic Clarity', {}).get('score', 'N/A'),
                r.get('ai_citation_potential', 'Unknown'),
                top_rec,
                '; '.join(r.get('strengths', [])[:2])
            ])

    print(f"\n💾 Saved {len(reports)} detailed reports to '{filename}'")


def save_html_report(reports: List[Dict], domain_context: Optional[Dict] = None,
                     filename: str = "ai_seo_report.html"):
    from urllib.parse import urlparse

    scores = [r.get('overall_score', 0) for r in reports if isinstance(r.get('overall_score'), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    all_recs = []
    for r in reports:
        page_url = r.get('url', '')
        for rec in r.get('recommendations', []):
            all_recs.append({**rec, 'page_url': page_url})

    critical_recs = [r for r in all_recs if r.get('priority') == 'Critical']
    high_recs = [r for r in all_recs if r.get('priority') == 'High']
    medium_recs = [r for r in all_recs if r.get('priority') == 'Medium']
    low_recs = [r for r in all_recs if r.get('priority') == 'Low']

    category_avgs = {}
    for r in reports:
        for m in r.get('metrics', []):
            cat = m.get('category', '')
            score = m.get('score')
            if isinstance(score, (int, float)):
                category_avgs.setdefault(cat, []).append(score)
    category_avgs = {k: round(sum(v) / len(v), 1) for k, v in category_avgs.items()}

    def _score_color(s):
        if s >= 70: return "#22c55e"
        if s >= 40: return "#eab308"
        return "#ef4444"

    def _score_bg(s):
        if s >= 70: return "#f0fdf4"
        if s >= 40: return "#fefce8"
        return "#fef2f2"

    def _status_badge(status):
        colors = {"Pass": ("#166534", "#dcfce7"), "Warning": ("#854d0e", "#fef9c3"), "Fail": ("#991b1b", "#fee2e2")}
        fg, bg = colors.get(status, ("#374151", "#f3f4f6"))
        return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600">{status}</span>'

    def _priority_badge(priority):
        colors = {"Critical": ("#991b1b", "#fee2e2"), "High": ("#854d0e", "#fef9c3"),
                  "Medium": ("#1e40af", "#dbeafe"), "Low": ("#374151", "#f3f4f6")}
        fg, bg = colors.get(priority, ("#374151", "#f3f4f6"))
        return f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">{priority}</span>'

    domain_name = ""
    if reports:
        first_url = reports[0].get('url', '')
        parsed = urlparse(first_url)
        domain_name = parsed.netloc

    metric_cards_html = ""
    for cat, avg in category_avgs.items():
        color = _score_color(avg)
        bg = _score_bg(avg)
        metric_cards_html += f'''
        <div style="background:{bg};border:1px solid {color}33;border-radius:12px;padding:16px;text-align:center;min-width:140px">
            <div style="font-size:28px;font-weight:700;color:{color}">{avg}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:4px">{cat}</div>
        </div>'''

    domain_html = ""
    if domain_context:
        items = [
            ("Robots.txt", domain_context.get('robots_summary', 'N/A')),
            ("llms.txt", domain_context.get('llms_txt', 'N/A')),
            ("Sitemap", domain_context.get('sitemap_summary', 'N/A')),
            ("AI Discovery", domain_context.get('ai_files_summary', 'N/A')),
        ]
        domain_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px">'
        for label, value in items:
            is_good = value in ("Present",) or ("0 of" in str(value) and "blocked" in str(value))
            icon = "✅" if is_good else "⚠️"
            domain_html += f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px"><span style="font-size:13px;color:#6b7280">{label}</span><br><span style="font-size:14px;font-weight:500">{icon} {value}</span></div>'
        domain_html += '</div>'

    pages_rows = ""
    for i, page in enumerate(reports, 1):
        score = page.get('overall_score', 0)
        color = _score_color(score)
        readiness = page.get('ai_readiness', 'Unknown')
        title = page.get('title', 'Untitled')[:60]
        url = page.get('url', '')

        metrics_cells = ""
        for m in page.get('metrics', []):
            ms = m.get('score', 'N/A')
            if isinstance(ms, (int, float)):
                mc = _score_color(ms)
                metrics_cells += f'<td style="text-align:center"><span style="color:{mc};font-weight:600">{ms}</span></td>'
            else:
                metrics_cells += f'<td style="text-align:center;color:#9ca3af">{ms}</td>'

        strengths_html = ""
        for s in page.get('strengths', [])[:3]:
            strengths_html += f'<li style="font-size:13px;color:#374151">{s}</li>'

        page_recs_html = ""
        page_recs = page.get('recommendations', [])
        for rec in page_recs[:5]:
            page_recs_html += f'<div style="padding:6px 0;border-bottom:1px solid #f3f4f6">{_priority_badge(rec.get("priority",""))} <span style="font-size:13px">{rec.get("action","")}</span>'
            if rec.get('impact'):
                page_recs_html += f'<br><span style="font-size:12px;color:#6b7280;margin-left:60px">Impact: {rec["impact"]}</span>'
            page_recs_html += '</div>'

        detail_metrics_html = ""
        for m in page.get('metrics', []):
            ms = m.get('score', 'N/A')
            bar_width = ms if isinstance(ms, (int, float)) else 0
            bar_color = _score_color(bar_width) if isinstance(ms, (int, float)) else "#d1d5db"
            detail_metrics_html += f'''
            <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #f3f4f6">
                <div style="width:160px;font-size:13px;color:#374151">{m.get('category','')}</div>
                <div style="flex:1;background:#f3f4f6;border-radius:4px;height:20px;overflow:hidden">
                    <div style="width:{bar_width}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.3s"></div>
                </div>
                <div style="width:35px;text-align:right;font-weight:600;color:{bar_color};font-size:13px">{ms}</div>
                {_status_badge(m.get('status',''))}
            </div>'''

        pages_rows += f'''
        <div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin-bottom:16px" id="page-{i}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <div>
                    <h3 style="margin:0;font-size:16px;color:#111827">{title}</h3>
                    <a href="{url}" style="font-size:13px;color:#6b7280;text-decoration:none" target="_blank">{url}</a>
                </div>
                <div style="text-align:center">
                    <div style="font-size:36px;font-weight:700;color:{color}">{score}</div>
                    <div style="font-size:12px;color:#6b7280">{readiness}</div>
                </div>
            </div>
            <details style="margin-top:8px">
                <summary style="cursor:pointer;font-weight:600;font-size:14px;color:#4b5563;padding:8px 0">Show detailed metrics & recommendations</summary>
                <div style="margin-top:12px">
                    <h4 style="margin:12px 0 8px;font-size:14px;color:#374151">Metric Breakdown</h4>
                    {detail_metrics_html}
                    <div style="display:flex;gap:24px;margin-top:16px">
                        <div style="flex:1">
                            <h4 style="margin:0 0 8px;font-size:14px;color:#374151">Strengths</h4>
                            <ul style="margin:0;padding-left:20px">{strengths_html or '<li style="color:#9ca3af">None identified</li>'}</ul>
                        </div>
                        <div style="flex:2">
                            <h4 style="margin:0 0 8px;font-size:14px;color:#374151">Recommendations</h4>
                            {page_recs_html or '<div style="color:#9ca3af;font-size:13px">No recommendations</div>'}
                        </div>
                    </div>
                </div>
            </details>
        </div>'''

    action_plan_html = ""
    for priority, recs, color_cls in [("Critical", critical_recs, "#fee2e2"),
                                       ("High", high_recs, "#fef9c3"),
                                       ("Medium", medium_recs, "#dbeafe"),
                                       ("Low", low_recs, "#f3f4f6")]:
        if not recs:
            continue
        action_plan_html += f'<h3 style="margin:16px 0 8px;font-size:15px">{_priority_badge(priority)} {len(recs)} {priority} Issues</h3>'
        for rec in recs[:15]:
            short_url = urlparse(rec.get('page_url', '')).path or '/'
            action_plan_html += f'''
            <div style="background:{color_cls};border-radius:8px;padding:10px 14px;margin-bottom:6px;font-size:13px">
                <strong>{rec.get('action','')}</strong>
                <span style="color:#6b7280;margin-left:8px">{rec.get('category','')}</span>
                <span style="color:#9ca3af;margin-left:8px;font-size:12px">{short_url}</span>
                {f'<br><span style="color:#6b7280;font-size:12px">Impact: {rec["impact"]}</span>' if rec.get('impact') else ''}
            </div>'''

    gauge_pct = avg_score / 100
    gauge_dash = 251.2 * gauge_pct
    gauge_color = _score_color(avg_score)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI SEO Audit Report - {domain_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; color: #111827; line-height: 1.5; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 24px; font-weight: 700; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 24px 0 12px; color: #374151; }}
  a {{ color: #2563eb; }}
  details > summary {{ list-style: none; }}
  details > summary::-webkit-details-marker {{ display: none; }}
  details > summary::before {{ content: "\\25B6 "; font-size: 10px; transition: transform 0.2s; display: inline-block; margin-right: 4px; }}
  details[open] > summary::before {{ transform: rotate(90deg); }}
  @media print {{
    details {{ open: true; }}
    details > summary {{ display: none; }}
    .no-print {{ display: none; }}
  }}
  @media (max-width: 768px) {{
    .metric-grid {{ grid-template-columns: repeat(2, 1fr) !important; }}
  }}
</style>
</head>
<body>
<div class="container">

<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;flex-wrap:wrap;gap:16px">
    <div>
        <h1>AI SEO Audit Report</h1>
        <p style="color:#6b7280;font-size:14px">{domain_name} &middot; {time.strftime("%B %d, %Y at %H:%M")} &middot; {len(reports)} page{"s" if len(reports) != 1 else ""} analyzed</p>
    </div>
    <div style="text-align:center">
        <svg width="100" height="100" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="none" stroke="#e5e7eb" stroke-width="8"/>
            <circle cx="50" cy="50" r="40" fill="none" stroke="{gauge_color}" stroke-width="8"
                    stroke-dasharray="{gauge_dash} 251.2" stroke-dashoffset="0"
                    transform="rotate(-90 50 50)" stroke-linecap="round"/>
            <text x="50" y="46" text-anchor="middle" font-size="22" font-weight="700" fill="{gauge_color}">{avg_score}</text>
            <text x="50" y="62" text-anchor="middle" font-size="10" fill="#6b7280">avg score</text>
        </svg>
    </div>
</div>

<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin-bottom:24px">
    <h2 style="margin:0 0 12px">Domain-Level Checks</h2>
    {domain_html or '<p style="color:#9ca3af">No domain context available</p>'}
</div>

<h2>Category Averages</h2>
<div class="metric-grid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:24px">
    {metric_cards_html}
</div>

<h2>Page Analysis</h2>
{pages_rows}

<div style="background:white;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin-top:24px">
    <h2 style="margin:0 0 8px">Action Plan ({len(all_recs)} total issues)</h2>
    <p style="font-size:13px;color:#6b7280;margin-bottom:12px">Prioritized recommendations across all pages. Address Critical items first.</p>
    {action_plan_html or '<p style="color:#22c55e;font-weight:500">No issues found!</p>'}
</div>

<div style="text-align:center;padding:24px;color:#9ca3af;font-size:12px">
    Generated by AI SEO Auditor v2.0.0 &middot; GEO/AEO Optimization Tool | UNU Campus Computing Centre
</div>

</div>
</body>
</html>'''

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"🌐 HTML report saved to '{filename}' — open in browser to view")


# --- 4. Main Crawler Logic ---
async def analyze_site(start_url: str):
    """Enhanced crawler with 2025 AI SEO checks"""
    print(f"🕵️  Initializing Enhanced AI SEO Crawler...")
    print(f"🎯 Target: {start_url}")
    
    # Extract base URL for llms.txt check
    from urllib.parse import urlparse
    parsed = urlparse(start_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        user_agent_mode="random",
        java_script_enabled=True,
        enable_stealth=True,
        ignore_https_errors=True,
    )

    deep_strategy = None
    max_depth = int(os.getenv("MAX_DEPTH", 1))
    max_pages = int(os.getenv("MAX_PAGES", 5))

    if max_depth > 0:
        deep_strategy = BFSDeepCrawlStrategy(max_depth=max_depth, max_pages=max_pages)

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        process_iframes=True,
        remove_overlay_elements=True,
        remove_consent_popups=True,
        delay_before_return_html=2.5,
        scan_full_page=True,
        wait_until="networkidle",
        magic=True,
        simulate_user=True,
        override_navigator=True,
        max_retries=0,
        deep_crawl_strategy=deep_strategy,
    )

    # Domain-level checks (run once before crawling)
    print(f"\n🔍 Running domain-level checks...")

    llms_txt_status = check_llms_txt(base_url)
    print(f"   {'✅' if llms_txt_status == 'Present' else '⚠️'} llms.txt: {llms_txt_status}")

    robots_data = check_robots_txt(base_url)
    robots_icon = "✅" if robots_data['found'] and robots_data['blocked_count'] == 0 else "⚠️"
    print(f"   {robots_icon} robots.txt: {robots_data['summary']}")

    sitemap_data = check_sitemap_xml(base_url)
    sitemap_icon = "✅" if sitemap_data['found'] else "⚠️"
    print(f"   {sitemap_icon} sitemap.xml: {sitemap_data['summary']}")

    ai_files_data = check_ai_discovery_files(base_url)
    print(f"   📄 AI discovery: {ai_files_data['summary']}")

    import requests as _req
    _http_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}

    def _is_empty_shell(html):
        if not html:
            return True
        stripped = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        stripped = re.sub(r'<style[^>]*>.*?</style>', '', stripped, flags=re.DOTALL | re.IGNORECASE)
        stripped = re.sub(r'<[^>]+>', '', stripped).strip()
        return len(stripped) < 100

    def _http_fallback(url):
        try:
            resp = _req.get(url, timeout=15, headers=_http_headers)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
        except Exception:
            pass
        return None

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(url=start_url, config=run_config)

        if not isinstance(results, list):
            results = [results]

        # If BFS was starved (browser blocked on first page), discover links via HTTP
        # and fetch additional pages manually
        if len(results) <= 1 and max_pages > 1:
            first_res = results[0] if results else None
            first_html = None
            if first_res and _is_empty_shell(first_res.html):
                first_html = _http_fallback(first_res.url)
            if first_html:
                first_res.html = first_html
                first_res.success = True
                first_res._markdown = None
                first_res._fallback_markdown = html_to_markdown_simple(first_html)
                first_res._fallback_links = extract_links_from_html(first_html, first_res.url)
                discovered = first_res._fallback_links.get('internal', [])
                seen_urls = {first_res.url.rstrip('/')}
                extra_urls = []
                for link in discovered:
                    href = link.get('href', '').split('#')[0].split('?')[0].rstrip('/')
                    if href and href not in seen_urls and href.startswith(base_url):
                        seen_urls.add(href)
                        extra_urls.append(href)
                        if len(extra_urls) >= max_pages - 1:
                            break

                if extra_urls:
                    print(f"\n🔄 Browser was blocked — fetching {len(extra_urls)} additional pages via HTTP...")
                    from crawl4ai.models import CrawlResult
                    for extra_url in extra_urls:
                        html = _http_fallback(extra_url)
                        if html and not _is_empty_shell(html):
                            fake_res = CrawlResult(url=extra_url, html=html, success=True)
                            fake_res._fallback_markdown = html_to_markdown_simple(html)
                            fake_res._fallback_links = extract_links_from_html(html, extra_url)
                            results.append(fake_res)

        print(f"\n✅ Crawl Complete: {len(results)} pages fetched")
        aggregated_report = []

        for idx, res in enumerate(results, 1):
            needs_fallback = not res.success or _is_empty_shell(res.html)

            if needs_fallback:
                print(f"\n🔎 [{idx}/{len(results)}] Analyzing: {res.url}")
                print(f"   ⚠️  Browser blocked, falling back to HTTP fetch...")
                fallback_html = _http_fallback(res.url)
                if fallback_html:
                    res.html = fallback_html
                    res.success = True
                    res._markdown = None
                    res._fallback_markdown = html_to_markdown_simple(fallback_html)
                    res._fallback_links = extract_links_from_html(fallback_html, res.url)
                    print(f"   ✅ HTTP fallback succeeded ({len(fallback_html):,} bytes)")
                else:
                    print(f"   ❌ HTTP fallback failed, skipping page")
                    continue
            else:
                print(f"\n🔎 [{idx}/{len(results)}] Analyzing: {res.url}")

            # Extract schema
            schema_data = extract_schema_tags(res.html)
            schema_icon = "✅" if schema_data['found'] else "❌"
            print(f"   {schema_icon} Schema: {', '.join(schema_data['types'][:3]) if schema_data['types'] else 'None found'}")

            # Extract structured content
            structured_content = extract_structured_content(res.html)
            print(f"   📊 Structure: Lists={structured_content['list_count']}, Tables={structured_content['table_count']}, FAQ={structured_content['has_faq']}")

            # Use fallback-generated data when available, otherwise crawl4ai data
            crawl_metadata = res.metadata if hasattr(res, 'metadata') else None
            crawl_links = getattr(res, '_fallback_links', None) or (res.links if hasattr(res, 'links') else None)
            page_markdown = getattr(res, '_fallback_markdown', None) or res.markdown

            meta_tags = extract_meta_tags(res.html, metadata=crawl_metadata)
            print(f"   🏷️  Meta: {meta_tags['summary']}")

            freshness = extract_content_freshness(res.html, metadata=crawl_metadata, schema_data=schema_data)
            print(f"   📅 Freshness: {freshness['summary']}")

            headings_data = analyze_heading_hierarchy(res.html)
            print(f"   📑 Headings: {headings_data['summary']}")

            link_data = analyze_links(crawl_links, res.url)
            print(f"   🔗 Links: {link_data['summary']}")

            eeat_signals = extract_eeat_signals(res.html, metadata=crawl_metadata, schema_data=schema_data)
            print(f"   👤 E-E-A-T: {eeat_signals['summary']}")

            content_quality = analyze_content_quality(page_markdown)
            print(f"   📝 Quality: {content_quality['summary']}")

            # Send to LLM
            print(f"   🧠 Analyzing with AI...")
            json_str = await analyze_content_with_llm(
                page_markdown,
                schema_data,
                structured_content,
                llms_txt_status,
                res.url,
                robots_data,
                meta_tags,
                freshness,
                headings_data,
                link_data,
                eeat_signals,
                content_quality,
                sitemap_data,
                ai_files_data,
            )
            
            if not json_str:
                continue

            try:
                data = json.loads(json_str)
                aggregated_report.append(data)
                score = data.get('overall_score', 0)
                readiness = data.get('ai_readiness', 'Unknown')
                print(f"   ✨ Score: {score}/100 | AI Readiness: {readiness}")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON Parse Error: {e}")

        domain_context = {
            "robots_summary": robots_data['summary'],
            "llms_txt": llms_txt_status,
            "sitemap_summary": sitemap_data['summary'],
            "ai_files_summary": ai_files_data['summary'],
        }
        return aggregated_report, domain_context

# --- 5. Enhanced Reporting ---
def print_detailed_report(reports: List[Dict], domain_context: Optional[Dict] = None):
    if not reports:
        print("\n❌ No valid reports generated.")
        return

    scores = [r.get('overall_score', 0) for r in reports if isinstance(r.get('overall_score'), (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0

    print("\n" + "="*80)
    print(f"🤖 AI SEO AUDIT REPORT - 2025/2026 STANDARDS")
    print(f"📊 Average Score: {avg_score:.1f}/100")
    print(f"📄 Pages Analyzed: {len(reports)}")
    print("="*80)

    if domain_context:
        print(f"\n🌐 DOMAIN-LEVEL FINDINGS:")
        print(f"   🤖 robots.txt: {domain_context.get('robots_summary', 'N/A')}")
        print(f"   📄 llms.txt: {domain_context.get('llms_txt', 'N/A')}")
        print(f"   🗺️  sitemap.xml: {domain_context.get('sitemap_summary', 'N/A')}")
        print(f"   🔍 AI discovery files: {domain_context.get('ai_files_summary', 'N/A')}")

    for i, page in enumerate(reports, 1):
        print(f"\n{'='*80}")
        print(f"[{i}] {page.get('title', 'Untitled')}")
        print(f"🔗 {page.get('url')}")
        print(f"📈 Overall Score: {page.get('overall_score')}/100")
        print(f"🎯 AI Readiness: {page.get('ai_readiness', 'Unknown')}")
        print(f"🎖️  Citation Potential: {page.get('ai_citation_potential', 'Unknown')}")

        print(f"\n💪 Strengths:")
        for strength in page.get('strengths', [])[:3]:
            print(f"   ✅ {strength}")

        print(f"\n📊 Detailed Metrics ({len(page.get('metrics', []))} categories):")
        for m in page.get('metrics', []):
            status_icon = {"Pass": "✅", "Warning": "⚠️", "Fail": "❌"}.get(m.get('status', ''), "❓")
            print(f"   {status_icon} {m.get('category')}: {m.get('score', 'N/A')}/100")
            print(f"      → {m.get('details')}")

        print(f"\n🎯 Priority Recommendations:")
        critical_recs = [r for r in page.get('recommendations', []) if r.get('priority') == 'Critical']
        high_recs = [r for r in page.get('recommendations', []) if r.get('priority') == 'High']

        for rec in (critical_recs + high_recs)[:5]:
            priority_icon = "🔴" if rec.get('priority') == 'Critical' else "🟡"
            print(f"   {priority_icon} [{rec.get('priority')}] {rec.get('action')}")
            if rec.get('impact'):
                print(f"      Impact: {rec.get('impact')}")

# --- 6. Main Execution ---
async def main():
    print("="*80)
    print("🚀 AI SEO AUDITOR 2025/2026 - GEO/AEO Optimization Tool")
    print("="*80)
    print("\n📋 This tool audits your site across 11 AI search categories:")
    print("   • Robots.txt AI crawler access (GPTBot, ClaudeBot, PerplexityBot, etc.)")
    print("   • Schema.org markup (JSON-LD) with date/author extraction")
    print("   • AI discovery files (llms.txt, llms-full.txt, ai.json)")
    print("   • OpenGraph & meta tags completeness")
    print("   • Content freshness signals (datePublished, dateModified)")
    print("   • AI-friendly content structure (Q&A, lists, tables, accordions)")
    print("   • Heading hierarchy validation (H1-H6 nesting, question headings)")
    print("   • Link profile (internal/external, authoritative citations)")
    print("   • E-E-A-T signals (author, credentials, fact-check attribution)")
    print("   • Content quality metrics (depth, statistics, quotable statements)")
    print("   • Semantic clarity and topic focus\n")

    target_url = input("🔗 Enter URL to analyze: ").strip()
    if not target_url.startswith("http"):
        target_url = "https://" + target_url

    reports, domain_context = await analyze_site(target_url)

    if reports:
        print_detailed_report(reports, domain_context=domain_context)
        save_to_csv(reports)
        save_html_report(reports, domain_context=domain_context)

        critical_issues = sum(
            1 for r in reports
            for rec in r.get('recommendations', [])
            if rec.get('priority') == 'Critical'
        )

        print(f"\n{'='*80}")
        print(f"📈 SUMMARY")
        print(f"{'='*80}")
        print(f"🔴 Critical Issues: {critical_issues}")
        print(f"💾 CSV report saved to: ai_seo_report_2025.csv")
        print(f"🌐 HTML dashboard saved to: ai_seo_report.html")
        print(f"\n🎯 Next Steps:")
        print(f"   1. Address Critical priority items first")
        print(f"   2. Unblock AI crawlers in robots.txt if blocked")
        print(f"   3. Implement missing schema markup (Article, FAQPage, Person)")
        print(f"   4. Add llms.txt and llms-full.txt if not present")
        print(f"   5. Add publication/modification dates to content")
        print(f"   6. Restructure content with question headings and direct answers")
        print("="*80)
    else:
        print("\n❌ No valid reports generated. Check your configuration.")

if __name__ == "__main__":
    asyncio.run(main())