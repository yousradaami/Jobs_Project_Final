
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "jobs_with_roles.csv",
    encoding="utf-8-sig"
)

print("=" * 60)
print("SKILLS ANALYSIS BY JOB ROLE")
print("=" * 60)

# Exclude non-technical and unclassified jobs
excluded_roles = [
    "Non-Technical / Business",
    "Other / Review"
]

df = df[~df["Job_Role"].isin(excluded_roles)].copy()

# ==========================================
# 2. EXTRACT SKILLS
# ==========================================

skill_records = []

for _, row in df.iterrows():

    role = row["Job_Role"]
    skills_text = str(row["Required_Skills"])

    if skills_text.lower() in ["nan", "none", ""]:
        continue

    skills = re.split(r"[,;|]", skills_text)

    # Count each skill only once per job
    unique_skills = set()

    for skill in skills:

        skill = skill.strip()

        if skill:
            unique_skills.add(skill)

    for skill in unique_skills:

        skill_records.append({
            "Job_Role": role,
            "Skill": skill
        })

# ==========================================
# 3. CALCULATE FREQUENCIES
# ==========================================

skills_df = pd.DataFrame(skill_records)

if skills_df.empty:
    print("No skills found in the dataset.")
    raise SystemExit

frequency = (
    skills_df.groupby(["Job_Role", "Skill"])
    .size()
    .reset_index(name="Frequency")
)

frequency = frequency.sort_values(
    ["Job_Role", "Frequency"],
    ascending=[True, False]
)

# Save complete results
frequency.to_csv(
    "skills_by_job_role.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 4. DISPLAY TOP 10 SKILLS
# ==========================================

print("\nTOP 10 SKILLS BY JOB ROLE")

for role in frequency["Job_Role"].unique():

    role_data = frequency[
        frequency["Job_Role"] == role
    ].head(10)

    print("\n" + "=" * 50)
    print(role)
    print("=" * 50)

    print(
        role_data[["Skill", "Frequency"]]
        .to_string(index=False)
    )

# ==========================================
# 5. CREATE VISUALIZATION FOLDER
# ==========================================

os.makedirs("visualizations", exist_ok=True)

# ==========================================
# 6. CREATE SEPARATE CHART FOR EACH ROLE
# ==========================================

role_counts = df["Job_Role"].value_counts()

# Select the 6 largest job categories
selected_roles = role_counts.head(6).index

for role in selected_roles:

    role_data = (
        frequency[frequency["Job_Role"] == role]
        .head(10)
        .sort_values("Frequency", ascending=True)
    )

    if role_data.empty:
        continue

    # Large figure for readability
    fig, ax = plt.subplots(figsize=(13, 8))

    fig.patch.set_facecolor("#FAF8F5")
    ax.set_facecolor("#FAF8F5")

    bars = ax.barh(
        role_data["Skill"],
        role_data["Frequency"],
        color="#75446F",
        height=0.60
    )

    # Chart title
    ax.set_title(
        f"TOP 10 MOST REQUESTED SKILLS\n{role}",
        fontsize=18,
        fontweight="bold",
        color="#3D263A",
        pad=25
    )

    # Axis labels
    ax.set_xlabel(
        "Number of Job Advertisements",
        fontsize=12,
        labelpad=12
    )

    ax.set_ylabel(
        "Required Skills",
        fontsize=12,
        labelpad=12
    )

    # Readable tick labels
    ax.tick_params(
        axis="y",
        labelsize=12,
        pad=10
    )

    ax.tick_params(
        axis="x",
        labelsize=11
    )

    # Show values at the end of bars
    ax.bar_label(
        bars,
        padding=8,
        fontsize=11,
        fontweight="bold",
        color="#3D263A"
    )

    # Grid
    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)

    # Remove unnecessary borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Spacing
    ax.margins(x=0.15)

    plt.tight_layout()

    # Safe filename
    filename = re.sub(
        r"[^a-z0-9]+",
        "_",
        role.lower()
    ).strip("_")

    # Save chart
    plt.savefig(
        f"visualizations/skills_{filename}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="#FAF8F5"
    )

    plt.close(fig)

    print(f"Chart created: {role}")

# ==========================================
# 7. FINAL MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("- skills_by_job_role.csv")
print("- 6 separate charts in visualizations folder")
