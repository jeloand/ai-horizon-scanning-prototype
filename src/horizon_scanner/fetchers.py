# horizon_scanner/fetchers.py
"""
Async helpers that download raw documents.

The functions are intentionally *stateless* – all secrets / limits are
passed in as arguments so fetchers.py has zero dependency on config files.
"""
from __future__ import annotations
import asyncio, logging, xml.etree.ElementTree as ET, re, json
from pathlib import Path
from typing import Dict, List

import httpx, feedparser
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# -------------------------------------------------
# generic helper with retry / back-off
# -------------------------------------------------
@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=10),
       reraise=True)
async def _get_with_retry(session: httpx.AsyncClient,
                          url: str,
                          **kw) -> httpx.Response:
    r = await session.get(url, **kw)
    r.raise_for_status()
    return r


# -------------------------------------------------
# individual fetchers
# -------------------------------------------------
def _clean_description(html: str) -> str:
    return re.sub(r"<[^<]+?>", "", html).strip()


async def fetch_rss(
    session: httpx.AsyncClient,
    source_name: str,
    url: str,
    timeout: int = 20,
) -> List[dict]:
    """Download one RSS/Atom feed and return a list of dicts."""
    try:
        r = await _get_with_retry(
            session,
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except Exception as exc:
        logger.warning("RSS fetch failed for %s → %s", source_name, exc)
        return []

    feed = feedparser.parse(r.text)
    return [
        {
            "title": e.get("title", ""),
            "description": _clean_description(e.get("summary", "")),
            "link": e.get("link", ""),
            "published": e.get("published", e.get("updated", "")),
            "source_name": source_name,
        }
        for e in feed.entries
    ]


async def fetch_scopus(
    session: httpx.AsyncClient,
    query: str,
    count: int,
    api_key: str,
) -> List[dict]:
    url = "https://api.elsevier.com/content/search/scopus"
    params = {"query": query, "count": count, "sort": "relevance"}
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}

    try:
        r = await _get_with_retry(session, url, params=params, headers=headers, timeout=30)
    except httpx.HTTPStatusError as exc:
        logger.warning("Scopus API error %s → %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logger.warning("Scopus request failed → %s", exc)
        return []

    data = r.json()
    return [
        {
            "title":       res.get("dc:title",        "No title"),
            "description": res.get("dc:description",  "No abstract available"),
            "link":        res.get("prism:url",       ""),
            "published":   res.get("prism:coverDate", ""),
            "source_name": "Scopus – Elsevier",
        }
        for res in data.get("search-results", {}).get("entry", [])
    ]


async def fetch_openaire(
    session: httpx.AsyncClient,
    query: str,
    max_results: int,
) -> List[dict]:
    url = "https://api.openaire.eu/search/publications"
    params = {"keywords": query, "size": max_results}

    try:
        r = await _get_with_retry(session, url, params=params, timeout=30)
    except httpx.HTTPStatusError as exc:
        logger.warning("OpenAIRE API error %s → %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logger.warning("OpenAIRE request failed → %s", exc)
        return []

    root = ET.fromstring(r.content)
    ns = {"oaf": "http://namespace.openaire.eu/oaf"}

    return [
        {
            "title":       res.findtext(".//oaf:title", "", ns).strip(),
            "description": res.findtext(".//oaf:description", "", ns).strip(),
            "link":        res.findtext(".//oaf:identifier", "", ns),
            "published":   res.findtext(".//oaf:dateofacceptance", "", ns),
            "source_name": "OpenAIRE – Publications",
        }
        for res in root.findall("oaf:result", ns)
    ]


# -------------------------------------------------
# public convenience wrapper
# -------------------------------------------------
async def gather_sources(                       # noqa: D401 – simple description
    rss_sources: Dict[str, str],
    scopus_query: str,
    scopus_count: int,
    scopus_key: str,
    openaire_query: str,
    openaire_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download all sources concurrently and return three DataFrames."""
    import pandas as pd  # local import to keep fetchers lightweight

    async with httpx.AsyncClient() as session:
        rss_tasks = [
            fetch_rss(session, name, url)
            for name, url in rss_sources.items()
        ]
        scopus_task = fetch_scopus(session, scopus_query, scopus_count, scopus_key)
        openaire_task = fetch_openaire(session, openaire_query, openaire_limit)

        rss_lists, scopus_list, openaire_list = await asyncio.gather(
            asyncio.gather(*rss_tasks),
            scopus_task,
            openaire_task,
        )

    rss_df = pd.DataFrame([item for sub in rss_lists for item in sub])
    return rss_df, pd.DataFrame(scopus_list), pd.DataFrame(openaire_list)
