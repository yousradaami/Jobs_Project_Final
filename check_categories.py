
import pandas as pd

df = pd.read_csv(
    "jobs_final_5_columns.csv",
    encoding="utf-8-sig"
)

print("Columns:", df.columns.tolist())
print("\nCategory examples:")

for value in df["Categories"].dropna().head(10):
    print("-", str(value)[:180])

print("\nTotal jobs:", len(df))
