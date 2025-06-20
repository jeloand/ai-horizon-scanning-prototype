# app_streamlit.py
# ---------------------------------------------------------------------
"""
Streamlit dashboard + chat front-end for the Horizon-Scanning prototype
-----------------------------------------------------------------------
• Overview tab   – KPI cards, source mix bar-chart, preview table
• Ask GPT-4o tab – RAG workflow (vector-search → GPT answer)
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# project-internal helpers ------------------------------------------------
from retrieval_backend import snippets, ready as index_ready          # vector search
from agent_app         import ask_llm                                 # GPT caller


# ─────────────────────────── Config & paths ──────────────────────────── #
CSV_PATH   = Path("horizon_scanning_combined.csv")
PARQUET    = Path("horizon_scanning_meta.parquet")
FAISS_FILE = Path("horizon_scanning.faiss")

st.set_page_config(page_title="Policy Horizon Scanner", layout="wide")

# --------------------------------------------------------------------- #
# 0.  Safety checks                                                     #
# --------------------------------------------------------------------- #
missing = [p.name for p in (CSV_PATH, PARQUET, FAISS_FILE) if not p.exists()]
if missing:
    st.error(
        "The following artefacts are missing:\n\n"
        + "\n".join(f"• {m}" for m in missing)
        + "\n\nPlease run `policy_signal_scanner_v3.py` first.",
        icon="🚨",
    )
    st.stop()
if not index_ready():
    st.error("Vector index not ready – cannot run retrieval.", icon="🚨")
    st.stop()


# --------------------------------------------------------------------- #
# 1.  Data loading (cached)                                             #
# --------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading CSV …")
def load_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, low_memory=False)

df = load_df()


# --------------------------------------------------------------------- #
# 2. Sidebar controls                                                   #
# --------------------------------------------------------------------- #
with st.sidebar:
    st.header("🔎  Filter")
    all_sources = df["source_name"].unique().tolist()
    sel_sources = st.multiselect(
        "Sources", options=all_sources, default=all_sources, key="src_sel"
    )

    max_n = int(df["trend_score"].rank(method="dense").max())
    top_n = st.slider("Top-N by trend score", 10, max_n, 50, step=10, key="topn")

    st.markdown("---")
    question = st.text_area("Ask a policy question", height=120, key="qbox")
    ask_btn  = st.button("🔍 Ask GPT-4o")


# --------------------------------------------------------------------- #
# 3.  Apply sidebar filters                                             #
# --------------------------------------------------------------------- #
flt = df[df["source_name"].isin(sel_sources)].nlargest(top_n, "trend_score")
# a timestamp we can use for caching GPT answers
index_stamp = int(FAISS_FILE.stat().st_mtime)


# --------------------------------------------------------------------- #
# 4.  Tabs                                                              #
# --------------------------------------------------------------------- #
tab_overview, tab_ask = st.tabs(["📊 Overview", "💬 Ask GPT-4o"])


# ===== Overview tab =================================================== #
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Items", len(flt))
    col2.metric("Unique sources", flt["source_name"].nunique())
    last_run = datetime.fromtimestamp(CSV_PATH.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    col3.metric("Last run", last_run)

    st.subheader("Items per source")
    st.bar_chart(flt["source_name"].value_counts().sort_values(ascending=False))

    st.subheader("Preview")
    st.dataframe(
        flt[
            [
                "title",
                "source_name",
                "recency_label",
                "trend_score",
                "pestle_tags",
            ]
        ].reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )


# ===== GPT-4o / RAG tab ============================================== #
def _build_prompt(ctx: str, q: str) -> str:
    return (
        "You are a concise labour-market policy assistant.\n\n"
        "### Context\n"
        f"{ctx}\n\n"
        "### Question\n"
        f"{q}\n\n"
        "### Answer (bullet points preferred):"
    )

@st.cache_data(show_spinner=False)
def cached_answer(question: str, stamp: int) -> dict[str, str]:
    """
    Cache key = (exact question string, timestamp of FAISS index).
    Returns: {'answer': str, 'context': str}
    """
    ctx = snippets(question, k=5)
    ans = ask_llm(_build_prompt(ctx, question))
    return {"answer": ans, "context": ctx}

with tab_ask:
    st.write("Ask a natural-language question about labour-market policy.")
    st.caption("The agent retrieves 5 most relevant snippets, then feeds them to GPT-4o.")

    if ask_btn and question.strip():
        with st.spinner("Retrieving + prompting GPT-4o …"):
            try:
                out = cached_answer(question.strip(), index_stamp)
                st.toast("Answer ready!", icon="✅")
            except Exception as e:
                st.exception(e)
                st.stop()

        with st.expander("🔎 Context snippets (top-5)"):
            st.markdown(out["context"])

        st.markdown("### GPT-4o says")
        st.markdown(out["answer"])
    elif ask_btn:
        st.warning("Please type a question before hitting *Ask*.", icon="⚠️")
