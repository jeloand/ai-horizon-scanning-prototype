import math
import pandas as pd
from policy_signal_scanner_v3 import _robust_date, _compute_trend

# --- tiny fixtures --------------------------------------------------
EXAMPLE_ROW = {
    "title": "EU employment policy update",
    "description": "New labour mobility directive discussed today",
    "link": "https://ec.europa.eu/news/2025/06/01/",
    "published": "1 Jun 2025",
    "days_old": 0,          # pretend it's brand-new
}

KEYWORDS = ["labour", "mobility", "policy"]

# --- tests ----------------------------------------------------------
def test_robust_date_parses_iso():
    """_robust_date returns an ISO-8601 string when given a nice date"""
    row = EXAMPLE_ROW.copy()
    assert _robust_date(row).startswith("2025-06-01")

def test_trend_score_positive():
    """Trend score should be > 0 when text contains keywords"""
    score = _compute_trend(EXAMPLE_ROW, KEYWORDS)
    assert score > 0

def test_trend_score_decays():
    """Older items get a smaller score (time-decay works)"""
    old = EXAMPLE_ROW.copy();      old["days_old"] = 100
    new = EXAMPLE_ROW.copy();      new["days_old"] = 1
    s_old = _compute_trend(old, KEYWORDS)
    s_new = _compute_trend(new, KEYWORDS)
    assert s_new > s_old

