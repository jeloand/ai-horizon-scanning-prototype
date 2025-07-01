import pandas as pd

meta = pd.read_parquet("horizon_scanning_meta.parquet")
print(meta.shape)          # rows, columns
print(meta.head(10))       # first 10 rows

# if you only want certain columns
print(meta[["title", "source_name", "recency_label"]].head())
