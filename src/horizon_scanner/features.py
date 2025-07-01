# horizon_scanner/features.py
import re, math, numpy as np, pandas as pd
from dateutil import parser as dparse
from typing import Sequence

# ------------------------------------------------------------------  
# 1. Robust date parser
# ------------------------------------------------------------------
_date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")    # YYYY-MM-DD in URL

def robust_date(row: pd.Series) -> str | None:
    """
    Parse RSS-style or free-text dates *or* fall back to a YYYY-MM-DD
    slug in the link.  
    Returns ISO string or None (which becomes NaT downstream).
    """
    pub = row["published"]

    # a) struct_time from feedparser
    if isinstance(pub, (tuple, list)):
        try:
            return pd.to_datetime(pub, utc=True).isoformat()
        except Exception:
            pass

    # b) free-text (RFC, ISO, “14 Feb 2025”, …)
    try:
        return dparse.parse(str(pub), fuzzy=True).isoformat()
    except (ValueError, TypeError, OverflowError):
        pass

    # c) fallback → slug in the URL
    m = _date_pat.search(row["link"])
    return m.group(1) if m else None


# ------------------------------------------------------------------  
# 2. Vectorised trend-score
# ------------------------------------------------------------------
def trend_score_vectorised(
    df: pd.DataFrame,
    policy_keywords: Sequence[str]
) -> pd.Series:
    """
    Compute the same “freq × time-decay” score for *every row* quickly.
    Returns a pandas Series sized like `df`.
    """

    # aggregate title + description once
    text_series = (
        df["title"].fillna("") + " " + df["description"].fillna("")
    ).str.lower()

    # total keyword frequency per row -----------------------------
    freq = sum(
        text_series.str.count(rf"\b{re.escape(k)}\b")  # word-boundary safe
        for k in policy_keywords
    )

    # exponential time-decay  (half-life ≈ 20 days)
    decay = np.exp(-df["days_old"] / 30)

    return freq * (1 + decay)
