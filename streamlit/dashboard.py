# dashboard.py  – run with:  streamlit run dashboard.py
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_PATH = Path("horizon_scanning_combined.csv")

st.set_page_config(page_title="Horizon-Scanning Dashboard", layout="wide")
st.title("🔭 Horizon-Scanning Source Mix")

# ---------- Safe, version-agnostic cache decorator -------------
if hasattr(st, "cache_data"):        # Streamlit ≥1.25
    _cache = st.cache_data
else:                                # legacy fallback
    _cache = st.cache

@_cache(show_spinner="Loading data …")
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV not found at **{path}** – run the scraper first.")
        st.stop()
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# ---------- Cards ------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Items per source")
    vc = df["source_name"].value_counts()
    st.bar_chart(vc)

with col2:
    st.subheader("🕒 Recency label split")
    rc = df["recency_label"].value_counts().sort_index()
    st.bar_chart(rc)

# ---------- Interactive filter ----------------------------------
st.divider()
st.subheader("🔍 Explore rows")
sources = st.multiselect(
    "Filter by source (empty = all)",
    options=sorted(df["source_name"].unique()),
    default=[],
)
subset = df if not sources else df[df["source_name"].isin(sources)]
st.dataframe(
    subset.head(100),
    use_container_width=True,
    hide_index=True,
)
if __name__ == "__main__":
    print("This file is a Streamlit app.\n"
          "Launch it with:\n\n    streamlit run dashboard.py\n")
