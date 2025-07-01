import os, re, json, yaml, math, logging, asyncio, argparse
from datetime import datetime
from pathlib import Path
from dateutil import parser as dparse

import httpx, feedparser, xml.etree.ElementTree as ET
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, Pipeline
import torch, numpy as np
from joblib import dump, load
from disk_cache import disk_memoize
from horizon_scanner.fetchers import gather_sources
from horizon_scanner.features import robust_date, trend_score_vectorised

# ------------------------------------------------------------------
# Cache cleanup helper – removes files older than N days
# ------------------------------------------------------------------
import time, glob

def clean_cache(dir_path=".cache", max_age_days=7):
    """
    Delete any files under `dir_path` older than `max_age_days`.
    Runs at the start of each pipeline execution.
    """
    if not os.path.isdir(dir_path):
        return  # nothing to clean

    now = time.time()
    cutoff = max_age_days * 86400  # seconds

    for fp in glob.glob(f"{dir_path}/**/*", recursive=True):
        try:
            if os.path.isfile(fp) and now - os.path.getmtime(fp) > cutoff:
                os.remove(fp)
        except OSError:
            pass
# ------------------------------------------------------------------

logger = logging.getLogger(__name__)       # becomes “horizon_scanner”

# ------------------------------------------------------------------#
#  Helper functions exposed for unit-tests                           #
# ------------------------------------------------------------------#
DATE_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})")   # YYYY-MM-DD in URL slug

def _robust_date(row: dict) -> str | None:
    """
    Parse many fuzzy date formats.
    Returns ISO-8601 string or None.
    """
    val = row.get("published", "")
    # 1) feedparser may give struct_time
    if isinstance(val, (tuple, list)):
        try:
            return pd.to_datetime(val, utc=True).isoformat()
        except Exception:
            pass
    # 2) free-text parsing
    try:
        return dparse.parse(str(val), fuzzy=True).isoformat()
    except (ValueError, TypeError, OverflowError):
        pass
    # 3) fallback: look for YYYY-MM-DD in the URL
    m = DATE_PAT.search(row.get("link", ""))
    return m.group(1) if m else None

def _compute_trend(row: pd.Series, keywords: list[str]) -> float:
    """
    freq * (1+decay)  where
      freq  = keyword hits in title+desc
      decay = exp(-days_old / 30)
    """
    text = f"{row['title']} {row['description']}".lower()
    freq = sum(text.count(k) for k in keywords)
    decay = math.exp(-row["days_old"] / 30)
    return freq * (1 + decay)

async def main(enable_summary: bool, output_file: str, profile_name: str) -> None:
    """Run the complete ETL → embedding pipeline."""

    # ------------------------------------------------------------------
    # root logger config
    # ------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Clean .cache/ files older than 7 days  ← NEW
    clean_cache(".cache", max_age_days=7)

    # ------------------------------------------------------------------
    # 1. Secrets / API keys (config.json)
    # ------------------------------------------------------------------
    try:
        with open("config.json", encoding="utf-8") as f:
            _cfg = json.load(f)
        SCOPUS_API_KEY = _cfg.get("SCOPUS_API_KEY", "")
    except FileNotFoundError:
        SCOPUS_API_KEY = ""

    if not SCOPUS_API_KEY:
        raise SystemExit(
            "❌  SCOPUS_API_KEY is missing. Put it in config.json "
            'e.g. {"SCOPUS_API_KEY": "your-key"}'
        )

    # ------------------------------------------------------------------
    # 1b.  Tuning constants (config.yaml)
    # ------------------------------------------------------------------
    try:
        with open("config.yaml", encoding="utf-8") as fh:
            cfg_yaml = yaml.safe_load(fh)
    except FileNotFoundError:
        raise SystemExit(
            "❌  config.yaml missing – create it next to the script. Example:\n"
            "queries:\n"
            "  scopus: \"labour mobility OR social protection OR employment policy\"\n"
            "  openaire: \"labour mobility\"\n"
            "limits:\n"
            "  scopus: 25\n"
            "  openaire: 10"
        )

    # pull values from YAML
    SCOPUS_QUERY         = cfg_yaml["queries"]["scopus"]
    OPENAIRE_QUERY       = cfg_yaml["queries"]["openaire"]
    SCOPUS_COUNT         = cfg_yaml["limits"]["scopus"]
    OPENAIRE_MAX_RESULTS = cfg_yaml["limits"]["openaire"]

    # NEW — optional extras (use .get so it's not mandatory)
    EXTRA_KW       = cfg_yaml.get("focus_keywords", [])
    EXCLUDE_DOMAINS = cfg_yaml.get("exclude_domains", [])

    # ------------------------------------------------------------------
    # 2. Load scan profile (feeds + keywords)
    # ------------------------------------------------------------------
    profile_path = Path("profiles") / f"{profile_name}.yml"
    if not profile_path.exists():
        raise SystemExit(f"❌  profile '{profile_name}' not found in /profiles")

    with open(profile_path, encoding="utf-8") as fh:
        prof_cfg = yaml.safe_load(fh)

    rss_sources     = prof_cfg.get("feeds") or {}      # None → {}
    policy_keywords = prof_cfg.get("keywords") or []   # None → []

    logger.info(
        "Using scan profile: %s  • %d feeds  • %d keywords",
        profile_name, len(rss_sources), len(policy_keywords)
    )

    with open("current_profile.txt", "w", encoding="utf-8") as fh:
        fh.write(profile_name)
    
    if not rss_sources or not policy_keywords:
        raise SystemExit("❌  profile YAML must have non-empty 'feeds' and 'keywords'.")

    # ---------- runtime flags ----------
    OUTPUT_FILE    = output_file
    ENABLE_SUMMARY = enable_summary

    # ---------- kick-off download ----------------------------------
    logger.info("Fetching RSS + Scopus + OpenAIRE asynchronously …")

    rss_df, scopus_df, openaire_df = await gather_sources(
        rss_sources,
        SCOPUS_QUERY,
        SCOPUS_COUNT,
        SCOPUS_API_KEY,
        OPENAIRE_QUERY,
        OPENAIRE_MAX_RESULTS,
    )

    logger.info("All async sources fetched.")

    # ---- early-exit guard: did we actually receive any rows? ----
    if rss_df.empty and scopus_df.empty and openaire_df.empty:
        logger.error(
            "❌  No data returned from RSS, Scopus, or OpenAIRE. "
            "Check your internet connection or source URLs."
        )
        return           # stop the pipeline cleanly

    # ========== STEP 3: Load CORDIS projects from JSON ==========

    def load_cordis_projects(filepath: str = "cordis-h2020projects.json",
                             max_results: int = 20) -> pd.DataFrame:
        """Load a small sample of Horizon-2020 projects (local JSON)."""
        if not Path(filepath).exists():
            logger.warning("CORDIS file %s not found – skipping that source.", filepath)
            return pd.DataFrame()               # empty → downstream code still works

        logger.info("Loading CORDIS Horizon-2020 projects …")
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

        logger.info("Retrieved %d CORDIS entries", len(selected))
        return pd.DataFrame(selected)

    # call it
    cordis_df = load_cordis_projects()

    # ---------- STEP 3b: Combine all sources and remove duplicates ----------
    from stakeholder_ner import add_stakeholder_column

    df = pd.concat([rss_df, scopus_df, openaire_df, cordis_df],
                   ignore_index=True)
    df.drop_duplicates(subset=["link", "title"], inplace=True)
    logger.info("Combined sources: %d rows", len(df))
    df = add_stakeholder_column(df)    # ← NEW: populate 'stakeholders' list

    if EXCLUDE_DOMAINS:
        pat = "|".join(map(re.escape, EXCLUDE_DOMAINS))
        before = len(df)
        df = df[~df["link"].str.contains(pat, na=False)]
        logger.info("Dropped %d rows from excluded domains", before - len(df))

    # ---------- Fill / parse published dates (uses helper) ----------
    df["published"] = df.apply(robust_date, axis=1)

    # ========== STEP 4: Age & recency labels (vectorised) ==========

    import numpy as np                      # make sure NumPy is in scope

    # -- clean / standardise published strings ---------------------
    clean_pub = (
        df["published"]
          .astype(str)            # guard against floats / NA
          .str.strip()
          .str.replace(r"\bCEST\b", "+02:00", regex=True)
          .str.replace(r"\bCET\b",  "+01:00", regex=True)
    )

    # 1️⃣ parse all dates as UTC and drop tz-info so they’re tz-naïve
    pub_dates = (
        pd.to_datetime(clean_pub, errors="coerce", utc=True)
          .dt.tz_localize(None)
    )

    # 2️⃣ age in days (tz-naïve “now” in UTC)
    df["days_old"] = (pd.Timestamp.utcnow().replace(tzinfo=None) - pub_dates).dt.days
    df["days_old"] = df["days_old"].fillna(10_000)   # NaNs → very old

    # 3️⃣ recency buckets derived from days_old  (always in sync)
    bins   = [-float("inf"), 7, 30, 90, float("inf")]
    labels = ["Recent (≤ 7 days)", "Last 30 days", "Last 3 months", "Older"]

    age = df["days_old"].replace(10_000, np.nan)      # treat 10 000 as “unknown”
    df["recency_label"] = (
        pd.cut(age, bins=bins, labels=labels)
          .astype(object)                              # convert Categorical → str
          .fillna("Unknown")
    )

    logger.info("Recency label breakdown:\n%s", df["recency_label"].value_counts())

        # ---------- flag future-dated items ----------
    future_mask = df["days_old"] < 0
    df.loc[future_mask, "recency_label"] = "Upcoming (future)"
    df["days_old"] = df["days_old"].clip(lower=0)     # clamp negatives to 0

    # ========== STEP 5: Policy keyword filter ==========
    pattern = re.compile(
        r'\b(?:' + '|'.join(map(re.escape, policy_keywords)) + r')\b',
        re.I
    )

    logger.info("Filtering for policy-relevant articles …")
    mask = (
        df["title"].str.contains(pattern, na=False) |
        df["description"].str.contains(pattern, na=False)
    )
    df = df[mask]

    # ---- store which policy keywords each article matched -------------
    text_series = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()

    def find_matches(text: str) -> list[str]:
        return [kw for kw in policy_keywords if kw.lower() in text]

    df["matched_keywords"] = text_series.apply(find_matches)
    # -------------------------------------------------------------------

    logger.info("%d articles left after filtering", len(df))
    if df.empty:
        raise SystemExit("No articles matched policy keywords – exiting.")

    # ---------- STEP 6: Trend score (vectorised) ----------
    df["trend_score"] = trend_score_vectorised(df, policy_keywords)

    # log-scale (for plots)
    df["trend_score_log"] = np.log1p(df["trend_score"]).clip(lower=0)
    df = df.sort_values("trend_score", ascending=False)

    # ========== STEP 7: TF-IDF per-document keywords (CACHED) ==========

    CACHE_PATH = Path(".cache/tfidf_vec.joblib")
    CACHE_PATH.parent.mkdir(exist_ok=True)

    logger.info("Extracting per-document TF-IDF keywords …")

    corpus = (df["title"].fillna("") + " " + df["description"].fillna(""))

    # -- load or fit the vectorizer ------------------------------------
    if CACHE_PATH.exists():
        vectorizer = load(CACHE_PATH)
        X = vectorizer.transform(corpus)
        logger.info("✓ Loaded cached TF-IDF vocabulary (%d terms)",
                    len(vectorizer.get_feature_names_out()))
    else:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        X = vectorizer.fit_transform(corpus)
        dump(vectorizer, CACHE_PATH)
        logger.info("✓ Fitted and cached TF-IDF vectorizer (%d terms)",
                    len(vectorizer.get_feature_names_out()))

    terms = vectorizer.get_feature_names_out()

    def best_terms(row_vector, top_n: int = 5) -> str:
        """Return the top-N TF-IDF terms for one sparse row."""
        if row_vector.nnz == 0:
            return ""
        idxs, scores = row_vector.indices, row_vector.data
        top = sorted(zip(idxs, scores), key=lambda t: t[1], reverse=True)[:top_n]
        return ", ".join(terms[i] for i, _ in top)

    # **Now** X exists, so we can extract keywords safely
    df["extracted_keywords"] = [best_terms(X[i]) for i in range(X.shape[0])]

    # ------------------------------------------------------------------
    # NEW · Build a review subset for human moderation

    # 1️⃣  guarantee a unique integer ID for every row
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    REVIEW_TREND_THRESHOLD = 0.80          # adjust anytime
    RARITY_THRESHOLD       = 0.90          # keep or tweak later

    # 2️⃣ choose which column to count
    if "matched_keywords" in df.columns:            # NEW – variable length
        kw_col = "matched_keywords"                 # list[str]
    elif "extracted_keywords" in df.columns:        # fallback
        kw_col = "extracted_keywords"
    elif "matched_keywords" in df.columns:          # STEP-5 policy match (list[str])
        kw_col = "matched_keywords"
    else:
        kw_col = None                               # rarity will be 0

    # 3️⃣  compute keyword frequency + rarity score
    if kw_col:
        if kw_col == "extracted_keywords":          # "kw1, kw2, kw3"
            df["keyword_freq"] = (
                df[kw_col]
                  .str.split(",")                   # → list[str]
                  .apply(lambda xs: len(set(map(str.strip, xs))))
            )
        else:                                       # column is already list[str]
            df["keyword_freq"] = df[kw_col].apply(lambda xs: len(set(xs)))
        df["rarity_score"] = 1 / (df["keyword_freq"] + 1.0)
    else:
        df["keyword_freq"] = 0
        df["rarity_score"]  = 0.0

    # 4️⃣  build the moderation queue
    review_df = df[
        (df["trend_score"] >= REVIEW_TREND_THRESHOLD) |
        (df["rarity_score"] >= RARITY_THRESHOLD)
    ].copy()

    review_df["approved"] = None                    # analyst flag
    review_df.to_parquet("review_items.parquet", index=False)
    logger.info("Wrote %d rows to review_items.parquet", len(review_df))

    # ---------- STEP 8: PESTLE Tagging (zero-shot, cached) ----------
    
    import hashlib

    ZSHOT_CACHE = Path(".cache/zshot_results.jsonl")
    ZSHOT_CACHE.parent.mkdir(exist_ok=True)

    # ---- load cached labels  (hash → dict(labels, scores)) ----------
    zshot_cache: dict[str, dict] = {}
    if ZSHOT_CACHE.exists():
        with ZSHOT_CACHE.open("r", encoding="utf-8") as fh:
            for line in fh:
                h, res_json = line.rstrip("\n").split("\t", 1)
                zshot_cache[h] = json.loads(res_json)
        logger.info("Loaded %d cached zero-shot results", len(zshot_cache))

    # one global classifier – re-use it for every row
    try:
        zshot: Pipeline = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            device=0 if torch.cuda.is_available() else -1,
        )
    except Exception as exc:
        logger.warning("Zero-shot model failed to load (%s); falling back to keyword tags.", exc)
        zshot = None

    pestle_labels = ["Political", "Economic", "Social", "Technological", "Legal", "Environmental"]

    def _hash(text: str) -> str:
        """Short SHA-1 hash for stable cache keys."""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    # -----------------------------------------------------------------
    # When zero-shot is available, classify only uncached texts
    # -----------------------------------------------------------------
    if zshot:
        texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()

        to_classify: list[str] = []
        idx_needing: list[int] = []       # row indices of to_classify
        cached_labels: list[str | None] = []

        for idx, txt in enumerate(texts):
            h = _hash(txt)
            if h in zshot_cache:
                cached = ", ".join(
                    lbl
                    for lbl, sc in zip(zshot_cache[h]["labels"], zshot_cache[h]["scores"])
                    if sc >= 0.30
                )
                cached_labels.append(cached)
            else:
                to_classify.append(txt)
                idx_needing.append(idx)
                cached_labels.append(None)          # placeholder

        # ---- call the model only if there are misses -----------------
        if to_classify:
            logger.info("Zero-shot: classifying %d new texts …", len(to_classify))

            def zshot_batch(items: list[str]) -> list[dict]:
                return zshot(items, pestle_labels, multi_label=True, batch_size=8)

            fresh = zshot_batch(to_classify)

            # write new results to cache file
            with ZSHOT_CACHE.open("a", encoding="utf-8") as fh:
                for txt, res in zip(to_classify, fresh):
                    h = _hash(txt)
                    zshot_cache[h] = res
                    fh.write(f"{h}\t{json.dumps(res, ensure_ascii=False)}\n")

            # fill the placeholders
            for idx, res in zip(idx_needing, fresh):
                cached_labels[idx] = ", ".join(
                    lbl for lbl, sc in zip(res["labels"], res["scores"]) if sc >= 0.30
                )

        df["pestle_tags"] = cached_labels
    else:
        logger.info("Zero-shot disabled; using empty tags for now.")
        df["pestle_tags"] = ""

    # ---------------- keyword fallback for empty rows ------------------------
    empty_mask = df["pestle_tags"] == ""
    if empty_mask.any():
        logger.info("Applying keyword fallback for %d articles.", empty_mask.sum())
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
        logger.info("Summarising in batches …")
        sum_model = pipeline("summarization",
                             model="sshleifer/distilbart-cnn-12-6")  # ← constructor only

        def build_input(title: str, desc: str) -> str:
            text = f"{title}. {desc}".strip()
            return text[:1024] if text else ""

        inputs = [build_input(t, d) for t, d in zip(df["title"], df["description"])]

        logger.info("Summarising %d documents …", len(inputs))
        summaries = sum_model(inputs, batch_size=8, truncation=True)   # ← here

        df["summary"] = [s["summary_text"] if s else "" for s in summaries]
    else:
        logger.info("Summarisation disabled by config.")
        df["summary"] = ""

    # ---------- STEP 9b · Generate embeddings & save FAISS index ----------
    # 1️⃣  rebuild a single, consecutive id column (0 … N-1)
    df = df.reset_index(drop=True)          # discard any old row index
    df["id"] = np.arange(len(df))           # now len(id) == len(df) by definition

    # (optional) keep analyst approvals if the column already exists
    if "approved" not in df.columns:
        df["approved"] = False              # ensure the column is present

    # 2️⃣  build the list of texts to embed
    texts = (
        df["summary"].replace("", np.nan)   # prefer summary when available
          .fillna(df["description"])
          .tolist()
    )

    # 3️⃣  encode
    from embed_loader import get_embedder
    import faiss, numpy as np

    embed_model = get_embedder()
    emb = embed_model.encode(texts,
                             batch_size=64,
                             show_progress_bar=True,
                             convert_to_numpy=True)

    # 4️⃣  safety check (remove later if you like)
    assert emb.shape[0] == len(df) == df["id"].nunique(), "❌ id / vector mismatch"

    # 5️⃣  build FAISS index with explicit IDs
    dim   = emb.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    index.add_with_ids(
        emb.astype("float32"),
        df["id"].to_numpy(dtype="int64")
    )

    faiss.write_index(index, "horizon_scanning.faiss")
    df.to_parquet("horizon_scanning_meta.parquet", index=False)   # metadata
    logger.info("🧠 Embeddings saved: horizon_scanning.faiss and *_meta.parquet")

    # ========== STEP 10: Save ==========
    df.to_csv(OUTPUT_FILE, index=False)
    logger.info("✅ Combined output saved to %s (%d rows)", OUTPUT_FILE, len(df))

if __name__ == "__main__":
    import argparse, asyncio

    parser = argparse.ArgumentParser(
        description="AI-powered horizon-scanning pipeline")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="enable batching summaries (slower)",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        default="horizon_scanning_combined.csv",
        help="output CSV file name",
    )
    parser.add_argument(
    "--profile",
    default="global",
    help="name of scan profile YAML inside /profiles (default: global)",
    )
    args = parser.parse_args()

    asyncio.run(
    main(
        enable_summary=args.summary,
        output_file=args.out,
        profile_name=args.profile,
    )
)
