"""Sanity test for the ``search`` helper.

This file serves two purposes:

* When executed via ``pytest`` it defines ``test_search`` which verifies that
  the retrieval backend returns at least one result and contains the expected
  columns.
* When run directly (``python test_query.py``) it prints a nicely formatted
  preview table of the top five search hits.
"""

from faiss_search import search

EXPECTED_COLS = {"title", "source_name", "recency_label", "pestle_tags", "distance"}


def test_search() -> None:
    """Basic functional test for the ``search`` helper."""
    df = search("impact of labour mobility on EU policy", k=5)
    assert not df.empty, "search() returned no results"
    assert EXPECTED_COLS.issubset(df.columns)


if __name__ == "__main__":
    import sys
    from tabulate import tabulate

    query = sys.argv[1] if len(sys.argv) > 1 else "impact of labour mobility on EU policy"
    df = search(query, k=5)
    preview = df[[
        "title",
        "source_name",
        "recency_label",
        "pestle_tags",
        "distance",
    ]].head(5)
    print(tabulate(preview, headers="keys", tablefmt="github", showindex=False))

