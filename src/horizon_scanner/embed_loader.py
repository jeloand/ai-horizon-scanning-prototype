# embed_loader.py
# --------------------------------------------------
# Load the SentenceTransformer only once (lazy singleton)

from functools import lru_cache
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedder():
    """Returns the singleton SentenceTransformer model."""
    return SentenceTransformer(EMBED_MODEL_NAME)
