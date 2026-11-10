
import pandas as pd

df = pd.read_csv("jobs_with_roles.csv", encoding="utf-8-sig")

other = df[df["Job_Role"] == "Other / Review"]

print("Jobs needing manual review:", len(other))
print("\nJob titles:\n")

for i, title in enumerate(other["Job_Title"], 1):
    print(f"{i}. {title}")
