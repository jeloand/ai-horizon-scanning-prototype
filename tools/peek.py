import pandas as pd
from pathlib import Path

# Absolute path to the CSV (raw-string avoids back-slash issues)
csv_path = Path(r"C:\Users\rodri\OneDrive - Rodrigo Dal Borgo\Paralogosai\Python\horizon_scanning_combined.csv")

pd.set_option("display.width", 180)  # make console output wider
print(pd.read_csv(csv_path).head(10))

