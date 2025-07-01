# retrieval_backend.py
# --------------------------------------------------------------
import faiss, numpy as np, pandas as pd
from pathlib import Path
from embed_loader import get_embedder     # 🠔 singleton embedding loader

BASE_DIR   = Path(__file__).resolve().parent
_VECTORS   = BASE_DIR / "horizon_scanning.faiss"
_META      = BASE_DIR / "horizon_scanning_meta.parquet"

# -----------------------------------------------------------------
# Helper: are artefacts present?
# -----------------------------------------------------------------
def ready() -> bool:
    return _VECTORS.exists() and _META.exists()

# -----------------------------------------------------------------
# Lazy-load FAISS index & metadata only when available
# -----------------------------------------------------------------
if ready():
    index = faiss.read_index(str(_VECTORS))
    meta  = pd.read_parquet(_META).set_index("id")   # id column from pipeline
else:
    index = None
    meta  = None

# -----------------------------------------------------------------
# Embedding helper (singleton)
# -----------------------------------------------------------------
def _embed(text: str) -> np.ndarray:
    embed_model = get_embedder()                    # loads once, then cached
    return embed_model.encode([text],
                              convert_to_numpy=True).astype("float32")

# -----------------------------------------------------------------
# Public: return formatted top-k snippets for a query
# -----------------------------------------------------------------
def snippets(query: str, k: int = 5) -> str:
    """
    Embed the query, retrieve k nearest documents,
    return them as a Markdown-formatted bullet list.
    """
    if not ready():
        return "_Vector index not found – run the pipeline first._"

    qv = _embed(query)
    D, I = index.search(qv, k=k)

    hits = meta.loc[I[0]].copy()
    hits["distance"] = D[0]

    lines = [
        f"- **{row.title}** ({row.source_name}) – {row.distance:.3f}\n  {row.description[:300]}…"
        for _, row in hits.iterrows()
    ]
    return "\n".join(lines)


