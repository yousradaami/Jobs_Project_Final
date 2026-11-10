
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# Read skills frequency data
df = pd.read_csv("skills_frequency_final.csv", encoding="utf-8-sig")

# Select Top 20 skills
top_skills = df.sort_values(
    by="Frequency",
    ascending=False
).head(20)

# Display results
print("=" * 55)
print("TOP 20 MOST REQUIRED IT SKILLS")
print("=" * 55)
print(top_skills.to_string(index=False))

# Create horizontal bar chart
plt.figure(figsize=(12, 8))

plt.barh(
    top_skills["Skill"][::-1],
    top_skills["Frequency"][::-1],
    color="#580F41"
)

plt.title(
    "Top 20 Most Required IT Skills",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Number of Job Advertisements")
plt.ylabel("Required Skills")

plt.tight_layout()

# Save chart
plt.savefig(
    "visualizations/top_20_skills.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nChart saved successfully!")
