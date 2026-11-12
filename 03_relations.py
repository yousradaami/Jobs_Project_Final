"""
03_relations.py  -  Phase 4: Relationship analysis

Questions answered
  A. Do required skills change with experience level?   (Junior / Mid / Senior)
  B. Do required skills change with job role?
  C. Is the job role related to the experience level?
  D. Which skills are asked together?                    (co-occurrence, lift)
  E. Is the number of skills related to the years of experience?
  F. Robustness: do the main results survive if the same job posted for several
     countries is counted only once?

Inputs  : jobs_clean_final.csv, skills_long.csv, skill_categories.csv
Outputs : rel_*.csv, relations_report.txt, visualizations/rel_*.png
"""
import io
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="white", font_scale=0.95)
os.makedirs("visualizations", exist_ok=True)

df = pd.read_csv("jobs_clean_final.csv")
cats = pd.read_csv("skill_categories.csv", index_col=0)["Category"]

report = io.StringIO()


def log(*args):
    print(*args)
    print(*args, file=report)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"visualizations/{name}", dpi=150)
    plt.close(fig)


def benjamini_hochberg(p):
    """Adjusted p-values (q-values) controlling the false discovery rate."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


# ---------------------------------------------------------------
# Binary job x skill matrix (jobs that list at least one skill)
# ---------------------------------------------------------------
jobs = df[df["Has_Skills"]].copy().reset_index(drop=True)
jobs["Skill_List"] = jobs["Required_Skills"].str.split(", ")
all_skills = sorted({s for lst in jobs["Skill_List"] for s in lst})
M = pd.DataFrame({s: jobs["Skill_List"].map(lambda lst, s=s: s in lst) for s in all_skills})

tech_order = [s for s in cats.index if cats[s] != "Methodology & General"]  # sorted by frequency
TOP = tech_order[:15]
LEVELS = ["Junior (0-2 yrs)", "Mid (3-4 yrs)", "Senior (5+ yrs)"]
MAIN_ROLES = ["Software Engineering", "Frontend / Web Development", "Full-Stack Development", "Backend Development"]

# ===============================================================
# A. SKILLS x EXPERIENCE
# ===============================================================
log("=" * 64)
log("A. SKILLS vs EXPERIENCE LEVEL (jobs with skills, Junior / Mid / Senior)")
log("=" * 64)
sub = jobs["Experience_Group"].isin(LEVELS)
log(f"Jobs compared: {sub.sum()}  (Internship n={int((jobs['Experience_Group'] == 'Internship').sum())} "
    f"and 'Not Specified' n={int((jobs['Experience_Group'] == 'Not Specified').sum())} are left out)")

rows = []
for s in TOP:
    present = M.loc[sub, s]
    grp = jobs.loc[sub, "Experience_Group"]
    table = pd.crosstab(grp, present).reindex(LEVELS)
    if True not in table.columns:
        continue
    chi2, p, dof, _ = stats.chi2_contingency(table)
    pct = (table[True] / table.sum(axis=1) * 100)
    rows.append({"Skill": s, "Category": cats[s],
                 **{f"%{lv.split(' ')[0]}": round(pct[lv], 1) for lv in LEVELS},
                 "Senior_minus_Junior_pp": round(pct[LEVELS[2]] - pct[LEVELS[0]], 1),
                 "Cramers_V": round(cramers_v(table), 3), "p_value": p})
tests = pd.DataFrame(rows)
tests["q_value_BH"] = benjamini_hochberg(tests["p_value"])
tests["Significant_5%"] = tests["q_value_BH"] < 0.05
tests = tests.sort_values("p_value")
tests.to_csv("rel_skill_experience_tests.csv", index=False, encoding="utf-8-sig")

show = tests.copy()
show["p_value"] = show["p_value"].map(lambda v: f"{v:.2e}")
show["q_value_BH"] = show["q_value_BH"].map(lambda v: f"{v:.4f}")
log("\nChi-square test per skill (is the share of jobs listing the skill the same across levels?)")
log(show.drop(columns="Category").to_string(index=False))
n_sig = int(tests["Significant_5%"].sum())
log(f"\n-> {n_sig} of {len(tests)} top skills differ significantly across levels (FDR 5%).")
log("   Cramer's V reads: <0.10 negligible, 0.10-0.30 small-to-moderate, >0.30 strong.")

# heatmap: % of jobs listing each skill, by experience group
all_levels = ["Internship"] + LEVELS + ["Not Specified"]
pct_exp = pd.DataFrame({lv: M[jobs["Experience_Group"] == lv].mean() * 100 for lv in all_levels}).loc[TOP]
pct_exp.round(1).to_csv("rel_skills_by_experience_pct.csv", encoding="utf-8-sig")
ns = jobs["Experience_Group"].value_counts()
fig, ax = plt.subplots(figsize=(8.5, 6.2))
sns.heatmap(pct_exp[LEVELS], annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "% of jobs in the group"}, ax=ax)
ax.set_xticklabels([f"{lv}\n(n={ns[lv]})" for lv in LEVELS], rotation=0)
ax.set(title="Share of jobs listing each skill, by experience level", ylabel="", xlabel="")
save(fig, "rel_skills_by_experience_heatmap.png")

# ===============================================================
# B. SKILLS x ROLE
# ===============================================================
log("\n" + "=" * 64)
log("B. SKILLS vs JOB ROLE (4 main roles; smaller roles have < 25 jobs each)")
log("=" * 64)
role_jobs = jobs["Job_Role"].isin(MAIN_ROLES)
pct_role = pd.DataFrame({r: M[jobs["Job_Role"] == r].mean() * 100 for r in MAIN_ROLES}).loc[TOP]
pct_role.round(1).to_csv("rel_skills_by_role_pct.csv", encoding="utf-8-sig")

rows = []
for s in TOP:
    table = pd.crosstab(jobs.loc[role_jobs, "Job_Role"], M.loc[role_jobs, s])
    chi2, p, _, _ = stats.chi2_contingency(table)
    rows.append({"Skill": s, "Cramers_V": round(cramers_v(table), 3), "p_value": p})
rt = pd.DataFrame(rows)
rt["q_value_BH"] = benjamini_hochberg(rt["p_value"])
rt = rt.sort_values("Cramers_V", ascending=False)
rt.to_csv("rel_skill_role_tests.csv", index=False, encoding="utf-8-sig")
log("\nSkills ranked by how strongly they depend on the role (Cramer's V):")
log(rt.assign(p_value=rt["p_value"].map(lambda v: f"{v:.2e}"), q_value_BH=rt["q_value_BH"].map(lambda v: f"{v:.2e}"))
    .to_string(index=False))
top_role = {}
for r in MAIN_ROLES:
    top_role[r] = pct_role[r].sort_values(ascending=False).head(5)
    log(f"\nTop 5 skills - {r}  (n={int((jobs['Job_Role'] == r).sum())}): "
        + ", ".join(f"{s} {v:.0f}%" for s, v in top_role[r].items()))

nr = jobs["Job_Role"].value_counts()
fig, ax = plt.subplots(figsize=(9.5, 6.2))
sns.heatmap(pct_role, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "% of jobs in the role"}, ax=ax)
ax.set_xticklabels([f"{r}\n(n={nr[r]})" for r in MAIN_ROLES], rotation=0, fontsize=8)
ax.set(title="Share of jobs listing each skill, by job role", ylabel="", xlabel="")
save(fig, "rel_skills_by_role_heatmap.png")

# ===============================================================
# C. ROLE x EXPERIENCE
# ===============================================================
log("\n" + "=" * 64)
log("C. JOB ROLE vs EXPERIENCE LEVEL (all jobs, Junior / Mid / Senior)")
log("=" * 64)
allsub = df[df["Experience_Group"].isin(LEVELS)].copy()
allsub["Role_Group"] = allsub["Job_Role"].where(allsub["Job_Role"].isin(MAIN_ROLES), "Other IT roles")
ct = pd.crosstab(allsub["Role_Group"], allsub["Experience_Group"])[LEVELS]
chi2, p, dof, expected = stats.chi2_contingency(ct)
log(ct.to_string())
log(f"\nChi-square = {chi2:.2f}, dof = {dof}, p = {p:.4g}, Cramer's V = {cramers_v(ct):.3f}")
log(f"Smallest expected count = {expected.min():.1f} (test is valid when >= 5)")
share = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
log("\nExperience mix inside each role (% of the role's jobs):")
log(share.to_string())
fig, ax = plt.subplots(figsize=(8.5, 4.8))
share.loc[MAIN_ROLES + ["Other IT roles"]].plot(kind="barh", stacked=True, ax=ax,
                                                 color=["#a8dadc", "#457b9d", "#1d3557"])
ax.set(title="Experience mix inside each role (Junior / Mid / Senior only)", xlabel="% of the role's jobs", ylabel="")
ax.legend(title="", loc="lower right", fontsize=8)
ax.invert_yaxis()
save(fig, "rel_role_vs_experience.png")

# ===============================================================
# D. CO-OCCURRENCE OF SKILLS
# ===============================================================
log("\n" + "=" * 64)
log("D. SKILLS ASKED TOGETHER (top 15 technical skills)")
log("=" * 64)
B = M[TOP].astype(int)
n = len(B)
support = B.mean()
co = B.T.dot(B) / n
lift_arr = (co.values / np.outer(support.values, support.values)).copy()
np.fill_diagonal(lift_arr, np.nan)
lift = pd.DataFrame(lift_arr, index=TOP, columns=TOP)

pairs = []
for a, b in itertools.combinations(TOP, 2):
    both = int((B[a] & B[b]).sum())
    pairs.append({"Skill_A": a, "Skill_B": b, "Jobs_with_both": both, "Lift": round(lift.loc[a, b], 2),
                  "P(B|A)_%": round(both / B[a].sum() * 100, 1), "P(A|B)_%": round(both / B[b].sum() * 100, 1)})
pairs = pd.DataFrame(pairs)
pairs.sort_values("Lift", ascending=False).to_csv("rel_skill_cooccurrence_pairs.csv", index=False, encoding="utf-8-sig")
solid = pairs[pairs["Jobs_with_both"] >= 80]
log("Lift > 1: asked together more often than chance; lift < 1: tend to exclude each other.")
log("(only pairs found together in >= 80 jobs)")
log("\nStrongest associations:")
log(solid.sort_values("Lift", ascending=False).head(8).to_string(index=False))
log("\nStrongest repulsions:")
log(solid.sort_values("Lift").head(6).to_string(index=False))

fig, ax = plt.subplots(figsize=(9.5, 8))
sns.heatmap(lift, annot=True, fmt=".1f", cmap="RdBu_r", center=1, vmin=0.2, vmax=3, annot_kws={"size": 7},
            cbar_kws={"label": "Lift (1 = independent)"}, ax=ax)
ax.set(title="Skill co-occurrence (lift) - top 15 technical skills")
save(fig, "rel_skill_cooccurrence_heatmap.png")

# ===============================================================
# E. NUMBER OF SKILLS vs YEARS OF EXPERIENCE
# ===============================================================
log("\n" + "=" * 64)
log("E. NUMBER OF SKILLS vs MINIMUM YEARS OF EXPERIENCE")
log("=" * 64)
num = jobs.dropna(subset=["Years_Min"])
rho, p = stats.spearmanr(num["Years_Min"], num["Skills_Count"])
log(f"Jobs with a numeric requirement: {len(num)}")
log(f"Spearman rho = {rho:.3f}, p = {p:.3g}")
log("-> " + ("negligible relationship" if abs(rho) < 0.1 else "weak relationship" if abs(rho) < 0.3 else "moderate or strong relationship"))
kw = stats.kruskal(*[g["Skills_Count"].values for _, g in jobs[jobs["Experience_Group"].isin(LEVELS)].groupby("Experience_Group")])
log(f"Kruskal-Wallis (skills per job across Junior/Mid/Senior): H = {kw.statistic:.2f}, p = {kw.pvalue:.3g}")

# ===============================================================
# F. ROBUSTNESS: count each (title, company) once
# ===============================================================
log("\n" + "=" * 64)
log("F. ROBUSTNESS - same job posted for several countries counted once")
log("=" * 64)
dedup_mask = ~jobs.duplicated(subset=["Job_Title", "Company"], keep="first")
log(f"Jobs with skills: {len(jobs)} -> after keeping one posting per (title, company): {int(dedup_mask.sum())}")
f_all = M[TOP].mean() * 100
f_ded = M.loc[dedup_mask, TOP].mean() * 100
rho2, _ = stats.spearmanr(f_all, f_ded)
cmp_ = pd.DataFrame({"All_postings_%": f_all.round(1), "One_per_job_%": f_ded.round(1)})
cmp_["Rank_all"] = f_all.rank(ascending=False).astype(int)
cmp_["Rank_dedup"] = f_ded.rank(ascending=False).astype(int)
cmp_.to_csv("rel_robustness_multi_country.csv", encoding="utf-8-sig")
log(cmp_.head(10).to_string())
log(f"\nSpearman correlation between the two rankings (top 15): {rho2:.3f}")
top5_same = set(cmp_.sort_values("Rank_all").head(5).index) == set(cmp_.sort_values("Rank_dedup").head(5).index)
log("Top-5 skills identical in both versions:", top5_same)

with open("relations_report.txt", "w", encoding="utf-8") as f:
    f.write(report.getvalue())
print("\nSaved: rel_*.csv, relations_report.txt, 4 charts rel_*.png in visualizations/")
