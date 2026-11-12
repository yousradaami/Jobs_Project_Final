"""
01_cleaning.py  -  Phase 2: Data Cleaning (Remote IT Job Market project)

Inputs  : jobs_with_roles.csv            (title, location, categories, experience, skills, role)
          jobs_informatique_cleaned.csv  (company, source, URL, date, IT relevance)
Outputs : jobs_clean_final.csv   -> THE single dataset used by every later script
          skills_long.csv        -> one row per (job, skill), ready for frequency analysis
          removed_rows.csv       -> rows excluded, with the reason (transparency)
          cleaning_report.txt    -> before / after numbers to copy into the report
"""
import io
import pandas as pd

ROLES_FILE = "jobs_with_roles.csv"
RAW_FILE = "jobs_informatique_cleaned.csv"

report = io.StringIO()


def log(*args):
    print(*args)
    print(*args, file=report)


# ---------------------------------------------------------------
# 1. LOAD + MERGE (both files have the same 2000 rows in the same order)
# ---------------------------------------------------------------
roles = pd.read_csv(ROLES_FILE)
raw = pd.read_csv(RAW_FILE)
assert len(roles) == len(raw) == 2000
assert (roles["Job_Title"].values == raw["Job_Title"].values).all(), "files are not aligned"

df = roles.rename(columns={"Categories": "Categories_Raw", "Job_Role": "Job_Role"})
for col in ["Job_ID", "Source", "Company", "Publication_Date", "Job_URL", "IT_Relevance"]:
    df[col] = raw[col].values

log("=" * 60)
log("BEFORE CLEANING")
log("=" * 60)
log("Rows:", len(df))
log("Missing values per column:")
log(df.isna().sum()[df.isna().sum() > 0].to_string())
log("Distinct experience labels:", df["Experience_Level"].nunique())
log("Distinct location values:", df["Location"].nunique())

# ---------------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------------
for col in ["Job_Title", "Location", "Company", "Required_Skills"]:
    df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

# ---------------------------------------------------------------
# 3. LOCATION: unify "unknown" values
# ---------------------------------------------------------------
df["Location"] = df["Location"].replace({"Remote": "Not Specified", "Unknown": "Not Specified", "": pd.NA})
df["Location"] = df["Location"].fillna("Not Specified")

# Some postings list many countries ("worldwide" jobs list up to ~250), and a few
# country names contain a comma. Protect those names, then count countries per job.
PROTECTED = {
    "Virgin Islands, British": "Virgin Islands (British)",
    "Virgin Islands, U.S.": "Virgin Islands (U.S.)",
    "Congo, The Democratic Republic of the": "Congo (Democratic Republic)",
    "Palestine, State of": "Palestine",
    "Saint Helena, Ascension and Tristan da Cunha": "Saint Helena",
    "Bonaire, Sint Eustatius and Saba": "Bonaire Sint Eustatius and Saba",
    "Micronesia, Federated States of": "Micronesia",
    "Korea, Republic of": "South Korea",
    "Taiwan, Province of China": "Taiwan",
    "Tanzania, United Republic of": "Tanzania",
}


def parse_countries(text):
    if text == "Not Specified":
        return []
    for old, new in PROTECTED.items():
        text = text.replace(old, new)
    return [c.strip() for c in text.split(",") if c.strip()]


_countries = df["Location"].map(parse_countries)
_bad = {"British", "U.S.", "State of", "Federated States of", "The Democratic Republic of the"}
assert not (set(c for lst in _countries for c in lst) & _bad), "comma-split country names remain"
df["Countries_Count"] = _countries.map(len)
df["Location_Type"] = pd.cut(df["Countries_Count"], bins=[-1, 0, 1, 10, 10_000],
                             labels=["Not Specified", "Single country", "Several countries (2-10)",
                                     "Worldwide (>10 countries)"]).astype(str)

# ---------------------------------------------------------------
# 4. EXPERIENCE LEVEL: 16 raw labels -> 5 groups (+ numeric minimum years)
# ---------------------------------------------------------------
EXPERIENCE_MAP = {
    "Internship": "Internship",
    "Entry Level": "Junior (0-2 yrs)",
    "Junior": "Junior (0-2 yrs)",
    "0 Years": "Junior (0-2 yrs)",
    "1+ Year": "Junior (0-2 yrs)",
    "1+ Years": "Junior (0-2 yrs)",
    "2 Years": "Junior (0-2 yrs)",
    "2+ Years": "Junior (0-2 yrs)",
    "Mid Level": "Mid (3-4 yrs)",
    "3 Years": "Mid (3-4 yrs)",
    "3+ Years": "Mid (3-4 yrs)",
    "4 Years": "Mid (3-4 yrs)",
    "4+ Years": "Mid (3-4 yrs)",
    "Senior": "Senior (5+ yrs)",
    "5+ Years": "Senior (5+ yrs)",
    "Not Specified": "Not Specified",
}
df["Experience_Raw"] = df["Experience_Level"]
df["Experience_Group"] = df["Experience_Level"].map(EXPERIENCE_MAP)
unmapped = df.loc[df["Experience_Group"].isna(), "Experience_Level"].unique()
assert len(unmapped) == 0, f"unmapped experience labels: {unmapped}"
df["Years_Min"] = pd.to_numeric(df["Experience_Raw"].str.extract(r"(\d+)")[0], errors="coerce")
df = df.drop(columns=["Experience_Level"])

# ---------------------------------------------------------------
# 5. SKILLS: split, strip, remove in-row duplicates, count
# ---------------------------------------------------------------
def split_skills(text):
    if pd.isna(text) or not str(text).strip():
        return []
    seen, out = set(), []
    for s in str(text).split(","):
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


skills_list = df["Required_Skills"].map(split_skills)
df["Skills_Count"] = skills_list.map(len)
df["Has_Skills"] = df["Skills_Count"] > 0
df["Required_Skills"] = skills_list.map(", ".join)

# ---------------------------------------------------------------
# 6. SAME TITLE + COMPANY = the same job posted for several countries.
#    They are NOT true duplicates (different location and URL), so we KEEP
#    them but flag them, so the analysis can test their effect.
# ---------------------------------------------------------------
key = ["Job_Title", "Company"]
df["Multi_Country_Posting"] = df.duplicated(subset=key, keep=False)
true_dups = df.duplicated(subset=["Job_URL"]).sum() + df.duplicated(subset=["Job_Title", "Company", "Location"]).sum()
log(f"\nTrue duplicates (same URL, or same title+company+location): {true_dups}")
df = df.drop_duplicates(subset=["Job_Title", "Company", "Location"]).copy()

# ---------------------------------------------------------------
# 7. REMOVE NON-IT ROWS (kept in removed_rows.csv with the reason)
# ---------------------------------------------------------------
reason = pd.Series("", index=df.index)
reason[df["IT_Relevance"] != "Relevant"] = "IT_Relevance = Needs review (ambiguous title)"
reason[df["Job_Role"] == "Non-Technical / Business"] = "Non-IT role"
removed = df[reason != ""].assign(Removal_Reason=reason[reason != ""])
df = df[reason == ""].copy()
removed.to_csv("removed_rows.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------
# 8. FINAL DATASET + SKILLS TABLE
# ---------------------------------------------------------------
df = df.reset_index(drop=True)
df.insert(0, "Job_Number", df.index + 1)
final_cols = ["Job_Number", "Job_ID", "Source", "Job_Title", "Company", "Location", "Publication_Date",
              "Countries_Count", "Location_Type",
              "Job_Role", "Experience_Raw", "Experience_Group", "Years_Min",
              "Required_Skills", "Skills_Count", "Has_Skills", "Multi_Country_Posting",
              "Categories_Raw", "Job_URL"]
df = df[final_cols]
df.to_csv("jobs_clean_final.csv", index=False, encoding="utf-8-sig")

skills_long = (df[df["Has_Skills"]][["Job_Number", "Job_Role", "Experience_Group", "Required_Skills"]]
               .assign(Skill=lambda x: x["Required_Skills"].str.split(", "))
               .explode("Skill")
               .drop(columns="Required_Skills")
               .reset_index(drop=True))
skills_long.to_csv("skills_long.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------
# 9. AFTER-CLEANING REPORT
# ---------------------------------------------------------------
log("\n" + "=" * 60)
log("AFTER CLEANING")
log("=" * 60)
log("Rows kept:", len(df), "| rows removed:", len(removed))
log(removed["Removal_Reason"].value_counts().to_string())
log("\nExperience groups:")
log(df["Experience_Group"].value_counts().to_string())
log(f"\nJobs without skills: {(~df['Has_Skills']).sum()} ({(~df['Has_Skills']).mean():.1%}) -> kept, excluded from skill statistics")
log(f"Jobs flagged Multi_Country_Posting: {df['Multi_Country_Posting'].sum()}")
log(f"Distinct skills: {skills_long['Skill'].nunique()} | skill records: {len(skills_long)}")
log("\nLocation type:")
log(df["Location_Type"].value_counts().to_string())
log("Remaining missing values:", int(df.drop(columns=["Categories_Raw"]).isna().sum().sum()))

with open("cleaning_report.txt", "w", encoding="utf-8") as f:
    f.write(report.getvalue())
print("\nSaved: jobs_clean_final.csv, skills_long.csv, removed_rows.csv, cleaning_report.txt")