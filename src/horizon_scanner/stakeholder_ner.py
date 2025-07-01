# stakeholder_ner.py
# ------------------------------------------------------------
"""
Lightweight helper that adds a `stakeholders` column (list[str])
to a pandas DataFrame containing 'title' and 'description'.
"""
from __future__ import annotations
import spacy, pandas as pd
from functools import lru_cache

# Load once; cached so repeated imports are instant.
# (CPU is fine; spaCy auto-detects GPU if available.)
@lru_cache(1)
def _nlp():
    return spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

def extract_stakeholders(text: str) -> list[str]:
    doc = _nlp()(text)
    # Keep only Person, Organisation, Geo-Political Entity
    ents = [ent.text.strip() for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE"}]
    # Remove duplicates but keep order
    return list(dict.fromkeys(ents))

def add_stakeholder_column(df: pd.DataFrame) -> pd.DataFrame:
    texts = (df["title"].fillna("") + ". " + df["description"].fillna(""))
    df["stakeholders"] = texts.apply(extract_stakeholders)
    return df
