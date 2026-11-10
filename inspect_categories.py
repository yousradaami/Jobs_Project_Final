
import pandas as pd

df = pd.read_csv(
    'jobs_final_5_columns.csv',
    encoding='utf-8-sig'
)

print("JOB CATEGORIES:")
print(df['Categories'].value_counts().to_string())

print("\nEXAMPLES:")
print(
    df[['Job_Title', 'Categories', 'Required_Skills']]
    .head(10)
    .to_string(index=False)
)
