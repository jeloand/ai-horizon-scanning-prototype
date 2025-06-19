# faiss_search.py
import faiss, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "horizon_scanning.faiss"
META_PATH  = BASE_DIR / "horizon_scanning_meta.parquet"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"          # must match the scraper

# --- load once at import ----------------------------------------
index = faiss.read_index(str(INDEX_PATH))
meta  = pd.read_parquet(META_PATH).set_index("id")   # id column = index
embed = SentenceTransformer(EMBED_MODEL_NAME)

# --- public helper ----------------------------------------------
def search(query: str, k: int = 5) -> pd.DataFrame:
    """Return a DataFrame with the k most similar articles to *query*."""
    vec = embed.encode([query]).astype("float32")
    D, I = index.search(vec, k)           # distances & IDs
    hits = meta.loc[I[0]].copy()          # look-up rows by id
    hits["distance"] = D[0]
    return hits.reset_index()

