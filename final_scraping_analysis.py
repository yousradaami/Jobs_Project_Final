
import pandas as pd
import re
import os
from collections import Counter

# ==================================================
# 1. LOAD DATASET
# ==================================================

INPUT_FILE = "jobs_informatique.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 65)
print("       FINAL JOB MARKET DATA ANALYSIS")
print("=" * 65)

print("\nOriginal dataset:", df.shape)


# ==================================================
# 2. EXTRACT EXPERIENCE LEVEL
# ==================================================

def extract_experience(description):

    text = str(description).lower()

    if text in ["nan", "none", ""]:
        return "Not Specified"

    # Explicit experience in years
    patterns = [
        (r'\b(?:5|6|7|8|9|10|[1-9]\d)\s*\+\s*years?\b',
         "5+ Years"),

        (r'\b4\s*\+\s*years?\b', "4+ Years"),
        (r'\b3\s*\+\s*years?\b', "3+ Years"),
        (r'\b2\s*\+\s*years?\b', "2+ Years"),
        (r'\b1\s*\+\s*years?\b', "1+ Year"),

        (r'\b(?:at least|minimum of)\s+(\d+)\s+years?\b',
         "EXPLICIT_MINIMUM"),
    ]

    for pattern, level in patterns:

        match = re.search(pattern, text)

        if match:

            if level == "EXPLICIT_MINIMUM":
                years = int(match.group(1))

                if years >= 5:
                    return "5+ Years"

                return f"{years}+ Years"

            return level

    # Exact experience requirements
    exact = re.search(
        r'\b(\d+)\s+years?\s+of\s+experience\b',
        text
    )

    if exact:
        years = int(exact.group(1))

        if years >= 5:
            return "5+ Years"

        return f"{years} Years"

    # Explicit experience levels
    if re.search(r'\bentry[- ]level\b|\bnew graduate\b|\brecent graduate\b', text):
        return "Entry Level"

    if re.search(r'\bintern(ship)?\b', text):
        return "Internship"

    if re.search(r'\bjunior\b', text):
        return "Junior"

    if re.search(r'\bmid[- ]level\b|\bintermediate level\b', text):
        return "Mid Level"

    if re.search(r'\bsenior\b|\blead level\b', text):
        return "Senior"

    if re.search(r'\bno experience required\b|\bwithout experience\b', text):
        return "No Experience Required"

    return "Not Specified"


# ==================================================
# 3. EXTRACT REQUIRED SKILLS
# ==================================================

SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript",
    "C++", "C#", "PHP", "Ruby", "Go", "Rust",
    "React", "Angular", "Vue.js", "Node.js",
    "HTML", "CSS", "Django", "Flask",
    "SQL", "MySQL", "PostgreSQL", "MongoDB",
    "Redis", "Firebase", "Oracle",
    "Power BI", "Tableau", "Excel",
    "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "Data Analysis",
    "Pandas", "NumPy", "TensorFlow", "PyTorch",
    "AWS", "Azure", "Google Cloud",
    "Docker", "Kubernetes", "Jenkins",
    "CI/CD", "Git", "GitHub",
    "Linux", "Windows Server",
    "Networking", "Cybersecurity",
    "REST API", "GraphQL",
    "Agile", "Scrum", "Software Engineering",
    "Spring Boot", ".NET", "Terraform"
]


def extract_skills(description):

    text = str(description).lower()

    if text in ["nan", "none", ""]:
        return ""

    found = []

    for skill in SKILLS:

        pattern = r'(?<!\w)' + re.escape(skill.lower()) + r'(?!\w)'

        if re.search(pattern, text):
            found.append(skill)

    return ", ".join(found)


# ==================================================
# 4. CREATE FINAL DATASET
# ==================================================

final_df = pd.DataFrame()

final_df["Job_Title"] = df["Job_Title"].fillna("Not Specified")

final_df["Location"] = df["Location"].fillna("Not Specified")

final_df["Categories"] = df["Categories"].fillna("Not Specified")

final_df["Experience_Level"] = df["Description"].apply(
    extract_experience
)

final_df["Required_Skills"] = df["Description"].apply(
    extract_skills
)


# ==================================================
# 5. SAVE FINAL DATASET
# ==================================================

OUTPUT_FILE = "jobs_final_5_columns.csv"

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 6. EXPERIENCE ANALYSIS
# ==================================================

print("\n" + "=" * 65)
print("EXPERIENCE LEVEL DISTRIBUTION")
print("=" * 65)

print(final_df["Experience_Level"].value_counts())


# ==================================================
# 7. SKILLS ANALYSIS
# ==================================================

counter = Counter()

for skills in final_df["Required_Skills"]:

    if skills:
        counter.update(skills.split(", "))

skills_df = pd.DataFrame(
    counter.most_common(),
    columns=["Skill", "Frequency"]
)

skills_df.to_csv(
    "skills_frequency_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTOP 20 REQUIRED SKILLS")

print(skills_df.head(20).to_string(index=False))


# ==================================================
# 8. EXPERIENCE AND SKILLS COMPARISON
# ==================================================

comparison = []

for level in final_df["Experience_Level"].unique():

    subset = final_df[
        final_df["Experience_Level"] == level
    ]

    level_counter = Counter()

    for skills in subset["Required_Skills"]:

        if skills:
            level_counter.update(skills.split(", "))

    for skill, frequency in level_counter.most_common(15):

        comparison.append({
            "Experience_Level": level,
            "Skill": skill,
            "Frequency": frequency
        })


comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    "skills_by_experience.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 9. FINAL REPORT
# ==================================================

print("\n" + "=" * 65)
print("FINAL DATASET SUMMARY")
print("=" * 65)

print("Total jobs:", len(final_df))

print("Number of columns:", len(final_df.columns))

print("\nColumns:")

print(final_df.columns.tolist())

print("\nMissing values:")

print(final_df.isnull().sum())

print("\nJobs without specified experience:")

print(
    (final_df["Experience_Level"] == "Not Specified").sum()
)

print("\nJobs without extracted skills:")

print(
    (final_df["Required_Skills"] == "").sum()
)

print("\nGenerated files:")

print("1.", OUTPUT_FILE)
print("2. skills_frequency_final.csv")
print("3. skills_by_experience.csv")

print("\nAnalysis completed successfully!")
 