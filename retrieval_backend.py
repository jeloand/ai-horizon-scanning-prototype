# retrieval_backend.py
# --------------------------------------------------------------
import faiss, numpy as np, pandas as pd
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_VECTORS = BASE_DIR / "horizon_scanning.faiss"
_META    = BASE_DIR / "horizon_scanning_meta.parquet"
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def ready() -> bool:
    return (_VECTORS.exists() and _META.exists())

index = faiss.read_index(str(_VECTORS)) if ready() else None
meta  = (pd.read_parquet(_META)
         .set_index("id")            # id column we added in STEP 9b
         if ready() else None)

def _embed(text: str) -> np.ndarray:
    return _EMBED_MODEL.encode([text], convert_to_numpy=True).astype("float32")

def snippets(query: str, k: int = 5) -> str:
    """Return *k* most-relevant documents as a single formatted string."""
    qv = _embed(query)
    D, I = index.search(qv, k=k)
    hits = meta.loc[I[0]].copy()
    hits["distance"] = D[0]
    out_lines = []
    for _, row in hits.iterrows():
        out_lines.append(f"- **{row.title}** ({row.source_name}) – {row.distance:.3f}\n  {row.description[:300]}…")
    return "\n".join(out_lines)

