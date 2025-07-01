# review_utils.py
import pandas as pd
from pathlib import Path

REVIEW_PATH  = Path("review_items.parquet")
MASTER_PATH  = Path("horizon_scanning_meta.parquet")

def merge_approved() -> None:
    """
    Copy analyst 'approved=True' flags from review_items.parquet
    back into horizon_scanning_meta.parquet, so the RAG agent
    and future sessions prefer vetted items.
    """
    if not (REVIEW_PATH.exists() and MASTER_PATH.exists()):
        return                                       # nothing to do

    review = pd.read_parquet(REVIEW_PATH)
    approved_ids = review.loc[review["approved"] == True, "id"].tolist()
    if not approved_ids:
        return

    df = pd.read_parquet(MASTER_PATH)
    if "approved" not in df.columns:
        df["approved"] = False

    df.loc[df["id"].isin(approved_ids), "approved"] = True
    df.to_parquet(MASTER_PATH, index=False)

