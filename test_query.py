from faiss_search import search

df = search("impact of labour mobility on EU policy", k=5)

# show a few useful columns
print(df[["title", "source_name", "recency_label", "pestle_tags", "distance"]]
        .to_markdown(index=False, tablefmt="github"))

