import os
import re
import json
import yaml
import math
import logging
import asyncio
from datetime import datetime

import httpx
import feedparser
import pandas as pd
import xml.etree.ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, Pipeline
import torch
from disk_cache import disk_memoize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------- load external RSS feed list ------------------------
try:
    with open("sources.yaml", encoding="utf-8") as fh:
        rss_sources: dict[str, str] = yaml.safe_load(fh)
except FileNotFoundError:
    raise SystemExit("❌  sources.yaml not found – put it in the same folder.")

# ---------------- load keyword list ----------------------------
with open("keywords.yaml", encoding="utf-8") as fh:
    cfg_kw = yaml.safe_load(fh)
policy_keywords: list[str] = cfg_kw["policy"]

# ========== CONFIG =============================================
SCOPUS_API_KEY = "API KEY HERE"
SCOPUS_QUERY   = "labour mobility OR social protection OR employment policy"
SCOPUS_COUNT   = 25

OPENAIRE_QUERY       = "labour mobility"
OPENAIRE_MAX_RESULTS = 10

OUTPUT_FILE   = "horizon_scanning_combined.csv"
ENABLE_SUMMARY = False           # flip to True to enable step 9

# ---------- helpers --------------------------------------------
def clean_description(desc: str) -> str:
    """Strip HTML tags from RSS <summary>."""
    return re.sub(r"<[^<]+?>", "", desc).strip()

@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=2, max=10),
       reraise=True)
async def get_with_retry(session: httpx.AsyncClient, url: str, **kw):
    resp = await session.get(url, **kw)
    resp.raise_for_status()
    return resp

# ---------- async fetchers -------------------------------------
async def fetch_rss(session: httpx.AsyncClient,
                    source_name: str,
                    url: str) -> list[dict]:
    try:
        r = await get_with_retry(session, url, timeout=20,
                                 headers={"User-Agent": "Mozilla/5.0"})
    except Exception as exc:
        logging.warning("RSS fetch failed for %s → %s", source_name, exc)
        return []

    feed = feedparser.parse(r.text)
    return [{
        "title":       e.get("title", ""),
        "description": clean_description(e.get("summary", "")),
        "link":        e.get("link", ""),
        "published":   e.get("published", e.get("updated", "")),
        "source_name": source_name,
    } for e in feed.entries]

async def fetch_scopus(session: httpx.AsyncClient) -> list[dict]:
    url     = "https://api.elsevier.com/content/search/scopus"
    params  = {"query": SCOPUS_QUERY, "count": SCOPUS_COUNT, "sort": "relevance"}
    headers = {"X-ELS-APIKey": SCOPUS_API_KEY, "Accept": "application/json"}
    try:
        r = await get_with_retry(session, url, params=params, headers=headers, timeout=30)
    except httpx.HTTPStatusError as exc:
        logging.warning("Scopus API error %s → %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logging.warning("Scopus request failed → %s", exc)
        return []
    data = r.json()
    return [{
        "title":       res.get("dc:title",           "No title"),
        "description": res.get("dc:description",     "No abstract available"),
        "link":        res.get("prism:url",          ""),
        "published":   res.get("prism:coverDate",    ""),
        "source_name": "Scopus – Elsevier",
    } for res in data.get("search-results", {}).get("entry", [])]

async def fetch_openaire(session: httpx.AsyncClient) -> list[dict]:
    url    = "https://api.openaire.eu/search/publications"
    params = {"keywords": OPENAIRE_QUERY, "size": OPENAIRE_MAX_RESULTS}
    try:
        r = await get_with_retry(session, url, params=params, timeout=30)
    except httpx.HTTPStatusError as exc:
        logging.warning("OpenAIRE API error %s → %s", exc.response.status_code, exc)
        return []
    except Exception as exc:
        logging.warning("OpenAIRE request failed → %s", exc)
        return []
    root = ET.fromstring(r.content)
    ns   = {"oaf": "http://namespace.openaire.eu/oaf"}
    return [{
        "title":       res.findtext(".//oaf:title", "", ns).strip(),
        "description": res.findtext(".//oaf:description", "", ns).strip(),
        "link":        res.findtext(".//oaf:identifier", "", ns),
        "published":   res.findtext(".//oaf:dateofacceptance", "", ns),
        "source_name": "OpenAIRE – Publications",
    } for res in root.findall("oaf:result", ns)]

# ---------- orchestrator ---------------------------------------
async def gather_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    async with httpx.AsyncClient() as session:
        rss_tasks     = [fetch_rss(session, n, u) for n, u in rss_sources.items()]
        scopus_task   = fetch_scopus(session)
        openaire_task = fetch_openaire(session)

        rss_lists, scopus_list, openaire_list = await asyncio.gather(
            asyncio.gather(*rss_tasks),
            scopus_task,
            openaire_task,
        )
        rss_df      = pd.DataFrame([item for sub in rss_lists for item in sub])
        scopus_df   = pd.DataFrame(scopus_list)
        openaire_df = pd.DataFrame(openaire_list)
        return rss_df, scopus_df, openaire_df

# ---------- kick off download ----------------------------------
logging.info("Fetching RSS + Scopus + OpenAIRE asynchronously …")
rss_df, scopus_df, openaire_df = asyncio.run(gather_sources())
logging.info("All async sources fetched.")

# ========== STEP 3: Load CORDIS projects from JSON ==========

from pathlib import Path   # already imported earlier, but keep if not

def load_cordis_projects(filepath: str = "cordis-h2020projects.json",
                         max_results: int = 20) -> pd.DataFrame:
    """Load a small sample of Horizon-2020 projects (local JSON)."""
    if not Path(filepath).exists():
        logging.warning("CORDIS file %s not found – skipping that source.", filepath)
        return pd.DataFrame()               # empty → downstream code still works

    logging.info("Loading CORDIS Horizon-2020 projects …")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    selected = []
    for rec in data:
        title = rec.get("title", "")
        objective = rec.get("objective", "")
        if not title and not objective:
            continue
        selected.append({
            "title":       title.strip(),
            "description": objective.strip(),
            "link":        f"https://cordis.europa.eu/project/id/{rec.get('rcn', '')}",
            "published":   rec.get("startDate", ""),   # fallback date
            "source_name": "CORDIS – H2020 Projects",
        })
        if len(selected) >= max_results:
            break

    logging.info("Retrieved %d CORDIS entries", len(selected))
    return pd.DataFrame(selected)

# call it
cordis_df = load_cordis_projects()

# ---------- STEP 3a: Combine all sources and remove duplicates ----------
df = pd.concat([rss_df, scopus_df, openaire_df, cordis_df],
               ignore_index=True)
df.drop_duplicates(subset=["link", "title"], inplace=True)
logging.info("Combined sources: %d rows", len(df))

# ---------- Fill missing published dates from URL slug ----------

date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")     # YYYY-MM-DD
def infer_date(row):
    if pd.notna(row["published"]) and row["published"].strip():
        return row["published"]                   # keep existing
    m = date_pat.search(row["link"])
    return m.group(1) if m else ""                # "" → handled later

df["published"] = df.apply(infer_date, axis=1)

# ========== STEP 4: Recency labels ==========

def assign_recency_label(pub_date_str: str) -> str:
    try:
        pub_date = pd.to_datetime(pub_date_str, errors="coerce")   # returns NaT on failure
        if pd.isna(pub_date):              # <-- handle empty or un-parsable dates
            return "Unknown"

        days_old = (pd.Timestamp.today() - pub_date).days
        if days_old <= 7:
            return "Recent (≤ 7 days)"
        if days_old <= 30:
            return "Last 30 days"
        if days_old <= 90:
            return "Last 3 months"
        return "Older"
    except Exception:
        return "Unknown"

df["recency_label"] = df["published"].apply(assign_recency_label)
logging.info("Recency label breakdown:\n%s", df["recency_label"].value_counts())

# ========== STEP 5: Policy keyword filter ==========

# non-capturing group + word boundaries → silences pandas warning
pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, policy_keywords)) + r')\b',
                     re.I)

logging.info("Filtering for policy-relevant articles …")
mask = (
    df["title"].str.contains(pattern, na=False) |
    df["description"].str.contains(pattern, na=False)
)
df = df[mask]

logging.info("%d articles left after filtering", len(df))
if df.empty:
    raise SystemExit("No articles matched policy keywords – exiting.")

# ========== STEP 6: Trend score (freq × time-decay) ==========

# -- clean / standardise published strings ---------------------
clean_pub = (df["published"]
             .astype(str)            # in case some values are float/NA
             .str.strip()            # remove leading/trailing spaces/new-lines
             .str.replace(r"\bCEST\b", "+02:00", regex=True)
             .str.replace(r"\bCET\b",  "+01:00", regex=True))

# 1️⃣ parse all dates as UTC, then drop the tz-info so they’re tz-naïve
pub_dates = (
    pd.to_datetime(clean_pub, errors="coerce", utc=True)
      .dt.tz_localize(None)
)

# 2️⃣ compare against a tz-naïve “now”
df["days_old"] = (pd.Timestamp.utcnow().replace(tzinfo=None) - pub_dates).dt.days
df["days_old"] = df["days_old"].fillna(10_000)   # NaNs → very old

def trend_score(row) -> float:
    text  = f"{row['title']} {row['description']}".lower()
    freq  = sum(text.count(k) for k in policy_keywords)
    decay = math.exp(-row["days_old"] / 30)       # half-life ≈ 20 days
    return freq * (1 + decay)

df["trend_score"] = df.apply(trend_score, axis=1)
df = df.sort_values("trend_score", ascending=False)

# ========== STEP 7: TF–IDF per‑document keywords (BUG‑FIX #2) ==========
logging.info("Extracting per‑document TF‑IDF keywords …")

corpus = (df["title"] + " " + df["description"]).fillna("")
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()


def best_terms(row_vector, top_n: int = 5) -> str:
    if row_vector.nnz == 0:
        return ""
    idxs = row_vector.indices
    scores = row_vector.data
    top = sorted(zip(idxs, scores), key=lambda t: t[1], reverse=True)[:top_n]
    return ", ".join(terms[i] for i, _ in top)


df["extracted_keywords"] = [best_terms(X[i]) for i in range(X.shape[0])]

# ========== STEP 8: PESTLE Tagging (zero-shot) ==========

# one global classifier – re-use it for every row
try:
    zshot: Pipeline = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",   # <-- valid ID
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as exc:
    logging.warning("Zero-shot model failed to load (%s); falling back to keyword tags.", exc)
    zshot = None

pestle_labels = ["Political", "Economic", "Social", "Technological", "Legal", "Environmental"]

# run the classifier in batches for speed ---------------------------------
@disk_memoize                      # ← one-liner cache
def zshot_batch(items: list[str]) -> list[dict]:
    """Call the zero-shot model once per run *or* load result from .cache/."""
    return zshot(items, pestle_labels, multi_label=True, batch_size=8)

if zshot:
    logging.info("Classifying PESTLE tags with zero-shot model (batched)… "
                 "(cached on disk)")
    texts   = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()
    results = zshot_batch(texts)    # ← cached call

    df["pestle_tags"] = [
        ", ".join(lbl for lbl, sc in zip(r["labels"], r["scores"]) if sc >= 0.30)
        for r in results
    ]
else:
    logging.info("Zero-shot disabled; using empty tags for now.")
    df["pestle_tags"] = ""

# ---------------- keyword fallback for empty rows ------------------------
empty_mask = df["pestle_tags"] == ""
if empty_mask.any():
    logging.info("Applying keyword fallback for %d articles.", empty_mask.sum())
    keyword_map = {
        "Political": ["regulation", "policy", "government", "parliament", "reform", "strategy"],
        "Economic": ["market", "employment", "jobs", "economy", "inflation", "budget", "trade"],
        "Social": ["inclusion", "equity", "migration", "gender", "welfare", "education", "community"],
        "Technological": ["ai", "digital", "automation", "technology", "platform", "innovation"],
        "Legal": ["law", "rights", "compliance", "litigation", "directive", "legal"],
        "Environmental": ["climate", "sustainability", "energy", "green", "carbon", "pollution"],
    }
    def keyword_tags(text: str) -> str:
        t = text.lower()
        return ", ".join(k for k, words in keyword_map.items() if any(w in t for w in words))
    df.loc[empty_mask, "pestle_tags"] = (
        df.loc[empty_mask, "title"].fillna("") + ". " + df.loc[empty_mask, "description"].fillna("")
    ).apply(keyword_tags)

# ========== STEP 9: Summarise (batched) ==========
if ENABLE_SUMMARY:
    logging.info("Summarising in batches …")
    sum_model = pipeline("summarization",
                         model="sshleifer/distilbart-cnn-12-6")  # ← constructor only

    def build_input(title: str, desc: str) -> str:
        text = f"{title}. {desc}".strip()
        return text[:1024] if text else ""

    inputs = [build_input(t, d) for t, d in zip(df["title"], df["description"])]

    logging.info("Summarising %d documents …", len(inputs))
    summaries = sum_model(inputs, batch_size=8, truncation=True)   # ← here

    df["summary"] = [s["summary_text"] if s else "" for s in summaries]
else:
    logging.info("Summarisation disabled by config.")
    df["summary"] = ""

# ---------- STEP 9b: Generate embeddings & save FAISS index ----------
df.reset_index(drop=False, inplace=True)          # creates column “index” (=row ID)
df.rename(columns={"index": "id"}, inplace=True)  # nicer column name

from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")   # ~80 MB

# use summary if present else description
texts = (df["summary"].replace("", np.nan)
         .fillna(df["description"])
         .tolist())

emb = embed_model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

# build FAISS index with explicit row-IDs (safer if rows are ever re-ordered)
dim   = emb.shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
index.add_with_ids(
    emb.astype("float32"),
    df["id"].to_numpy(dtype="int64")       # <-- use the real id column
)

faiss.write_index(index, "horizon_scanning.faiss")
df.to_parquet("horizon_scanning_meta.parquet", index=False)  # metadata alongside vectors
logging.info("🧠 Embeddings saved: horizon_scanning.faiss and *_meta.parquet")

# ========== STEP 10: Save ==========
df.to_csv(OUTPUT_FILE, index=False)
logging.info("✅ Combined output saved to %s (%d rows)", OUTPUT_FILE, len(df))
