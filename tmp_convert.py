import pandas as pd
import os

df = pd.read_csv("data/sales_data.csv")
df.to_parquet("data/sales_data.parquet", engine="pyarrow", index=False)
print(f"Compressed {os.path.getsize('data/sales_data.csv')/1e6:.1f}MB CSV to {os.path.getsize('data/sales_data.parquet')/1e6:.1f}MB Parquet!")
