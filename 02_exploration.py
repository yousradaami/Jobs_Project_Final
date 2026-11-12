"""
02_exploration.py  -  Phase 3: Data Exploration and Visualization

Inputs  : jobs_clean_final.csv, skills_long.csv   (produced by 01_cleaning.py)
Outputs : descriptive_stats.csv, skill_categories.csv, exploration_report.txt
          visualizations/eda_*.png  (6 charts)
"""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)
os.makedirs("visualizations", exist_ok=True)

df = pd.read_csv("jobs_clean_final.csv")
skills = pd.read_csv("skills_long.csv")

report = io.StringIO()


def log(*args):
    print(*args)
    print(*args, file=report)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"visualizations/{name}", dpi=150)
    plt.close(fig)


EXP_ORDER = ["Internship", "Junior (0-2 yrs)", "Mid (3-4 yrs)", "Senior (5+ yrs)", "Not Specified"]

# ---------------------------------------------------------------
# 1. GENERAL OVERVIEW
# ---------------------------------------------------------------
log("=" * 60)
log("1. DATASET OVERVIEW")
log("=" * 60)
log("Jobs:", len(df), "| columns:", df.shape[1])
log("Jobs with skills:", int(df["Has_Skills"].sum()), "| distinct skills:", skills["Skill"].nunique())
log("\nSource share (%):")
log((df["Source"].value_counts(normalize=True) * 100).round(1).to_string())

# ---------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS: number of skills per job
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("2. DESCRIPTIVE STATISTICS - skills per job (jobs with skills only)")
log("=" * 60)
ws = df[df["Has_Skills"]]


def describe(s):
    return pd.Series({"Jobs": len(s), "Mean": s.mean(), "Median": s.median(), "Std": s.std(),
                      "Min": s.min(), "Max": s.max()})


stats = pd.concat([describe(ws["Skills_Count"]).rename("All jobs").to_frame().T,
                   ws.groupby("Experience_Group")["Skills_Count"].apply(describe).unstack().reindex(EXP_ORDER)])
stats = stats.round(2)
stats.to_csv("descriptive_stats.csv", encoding="utf-8-sig")
log(stats.to_string())

log("\nSkills per job by role (roles with >= 30 jobs):")
by_role = ws.groupby("Job_Role")["Skills_Count"].agg(["count", "mean", "median"]).round(2)
log(by_role[by_role["count"] >= 30].sort_values("mean", ascending=False).to_string())

# ---------------------------------------------------------------
# 3. LOCATION: a job can list one country, several, or ~250 ("worldwide").
#    Counting countries over all rows would inflate every country, so the
#    country ranking uses single-country jobs only.
# ---------------------------------------------------------------
loc_type = df["Location_Type"].value_counts()
single = df[df["Location_Type"] == "Single country"]
country_counts = single["Location"].value_counts()
log("\n" + "=" * 60)
log("3. LOCATION")
log("=" * 60)
log("Location type:")
log(pd.concat([loc_type, (loc_type / len(df) * 100).round(1).rename("%")], axis=1).to_string())
log(f"\nDistinct countries among single-country jobs: {country_counts.shape[0]}")
log("Top 10 countries (single-country jobs only):")
log(country_counts.head(10).to_string())
log(f"Top 3 countries share of single-country jobs: {country_counts.head(3).sum() / len(single):.1%}")

# ---------------------------------------------------------------
# 4. SKILL CATEGORIES (so that generic items do not distort the ranking)
# ---------------------------------------------------------------
CATEGORIES = {
    "Languages & Web": ["JavaScript", "TypeScript", "Python", "Java", "Go", "C#", "PHP", "C++", "Ruby", "Rust",
                        "HTML", "CSS"],
    "Frameworks & APIs": ["React", "Angular", "Vue.js", "Node.js", ".NET", "Django", "Flask", "Spring Boot",
                          "GraphQL", "REST API"],
    "Cloud & DevOps": ["AWS", "Azure", "Google Cloud", "Kubernetes", "Docker", "Terraform", "Jenkins", "CI/CD",
                       "Git", "GitHub"],
    "Databases": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle", "Firebase"],
    "Systems, Network & Security": ["Linux", "Windows Server", "Networking", "Cybersecurity"],
    "Data & AI": ["Machine Learning", "Artificial Intelligence", "Deep Learning", "Data Analysis", "PyTorch",
                  "NumPy", "Pandas", "TensorFlow", "Power BI", "Tableau", "Excel"],
    "Methodology & General": ["Software Engineering", "Agile", "Scrum"],
}
skill_to_cat = {s: c for c, items in CATEGORIES.items() for s in items}
missing = set(skills["Skill"]) - set(skill_to_cat)
assert not missing, f"skills without a category: {missing}"
skills["Category"] = skills["Skill"].map(skill_to_cat)

freq = skills["Skill"].value_counts().rename("Frequency").to_frame()
freq["Category"] = freq.index.map(skill_to_cat)
freq["Share_of_jobs_with_skills_%"] = (freq["Frequency"] / len(ws) * 100).round(1)
freq.to_csv("skill_categories.csv", encoding="utf-8-sig")

log("\n" + "=" * 60)
log("4. SKILLS")
log("=" * 60)
log("Mentions by category (%):")
log((skills["Category"].value_counts(normalize=True) * 100).round(1).to_string())
tech = freq[freq["Category"] != "Methodology & General"]
log("\nTop 10 technical skills (% of jobs with skills):")
log(tech.head(10)[["Frequency", "Share_of_jobs_with_skills_%"]].to_string())

# ---------------------------------------------------------------
# 5. CHARTS
# ---------------------------------------------------------------
# 5.1 experience distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
vc = df["Experience_Group"].value_counts().reindex(EXP_ORDER)
sns.barplot(x=vc.index, y=vc.values, color="#3b6ea5", ax=ax)
for i, v in enumerate(vc.values):
    ax.text(i, v + 10, f"{v}\n({v / len(df):.0%})", ha="center", fontsize=9)
ax.set(title="Distribution of experience levels", xlabel="", ylabel="Number of jobs", ylim=(0, vc.max() * 1.18))
plt.setp(ax.get_xticklabels(), rotation=15)
save(fig, "eda_experience_distribution.png")

# 5.2 top 10 countries
fig, ax = plt.subplots(figsize=(8, 5))
top = country_counts.head(10).iloc[::-1]
ax.barh(top.index, top.values, color="#2a9d8f")
for i, v in enumerate(top.values):
    ax.text(v + 5, i, str(v), va="center", fontsize=9)
ax.set(title=f"Top 10 countries (single-country jobs, n={len(single)})", xlabel="Number of jobs")
ax.set_xlim(0, top.max() * 1.1)
save(fig, "eda_top_countries.png")

# 5.3 data sources
fig, ax = plt.subplots(figsize=(7, 4))
vc = df["Source"].value_counts()
sns.barplot(x=vc.index, y=vc.values, color="#e76f51", ax=ax)
for i, v in enumerate(vc.values):
    ax.text(i, v + 20, f"{v} ({v / len(df):.1%})", ha="center", fontsize=9)
ax.set(title="Jobs by data source", xlabel="", ylabel="Number of jobs", ylim=(0, vc.max() * 1.12))
save(fig, "eda_sources.png")

# 5.4 job roles
fig, ax = plt.subplots(figsize=(9, 5.5))
vc = df["Job_Role"].value_counts().iloc[::-1]
ax.barh(vc.index, vc.values, color="#6a4c93")
for i, v in enumerate(vc.values):
    ax.text(v + 8, i, str(v), va="center", fontsize=9)
ax.set(title="Distribution of job roles", xlabel="Number of jobs")
ax.set_xlim(0, vc.max() * 1.1)
save(fig, "eda_job_roles.png")

# 5.5 skills per job by experience (boxplot)
fig, ax = plt.subplots(figsize=(8.5, 5))
order = [e for e in EXP_ORDER if e != "Not Specified"] + ["Not Specified"]
sns.boxplot(data=ws, x="Experience_Group", y="Skills_Count", order=order, color="#a8dadc", ax=ax,
            flierprops={"markersize": 3})
ax.set(title="Number of required skills per job, by experience level", xlabel="", ylabel="Skills per job")
plt.setp(ax.get_xticklabels(), rotation=15)
save(fig, "eda_skills_per_job_by_experience.png")

# 5.6 top 15 technical skills, coloured by category
fig, ax = plt.subplots(figsize=(9, 6))
top15 = tech.head(15).iloc[::-1]
palette = dict(zip(CATEGORIES, sns.color_palette("Set2", len(CATEGORIES))))
ax.barh(top15.index, top15["Share_of_jobs_with_skills_%"], color=[palette[c] for c in top15["Category"]])
for i, v in enumerate(top15["Share_of_jobs_with_skills_%"]):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=9)
handles = [plt.Rectangle((0, 0), 1, 1, color=palette[c]) for c in CATEGORIES if c in set(top15["Category"])]
labels = [c for c in CATEGORIES if c in set(top15["Category"])]
ax.legend(handles, labels, loc="lower right", fontsize=8)
ax.set(title="Top 15 technical skills (generic items excluded)", xlabel="% of jobs that list the skill")
ax.set_xlim(0, top15["Share_of_jobs_with_skills_%"].max() * 1.12)
save(fig, "eda_top_technical_skills.png")

with open("exploration_report.txt", "w", encoding="utf-8") as f:
    f.write(report.getvalue())
print("\nSaved: descriptive_stats.csv, skill_categories.csv, exploration_report.txt, 6 charts in visualizations/")
