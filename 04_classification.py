"""
04_classification.py  -  Phase 5 (machine learning, part 1): supervised classification

Two prediction tasks, using ONLY the required skills as features:
  Task A  job role         (Software Engineering / Frontend / Full-Stack / Backend)
  Task B  experience level (Junior / Mid / Senior)

Why the evaluation is built this way
  * Several rows are the same job posted for different countries. If copies of the
    same job fall in both train and test, the score is inflated. We therefore split
    with groups = (job title, company): all copies of a job stay on the same side.
  * Every model is compared with a "predict the majority class" baseline.
  * A permutation test checks whether the model is better than chance.
  * Job_Role comes from keyword rules applied to the job TITLE (classify_job_roles.py),
    never from the skills, so the skills are not leaking the answer.

Inputs  : jobs_clean_final.csv
Outputs : ml_model_comparison.csv, ml_role_coefficients.csv, ml_classification_report.txt,
          visualizations/ml_*.png
"""
import io
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate

warnings.filterwarnings("ignore")
sns.set_theme(style="white", font_scale=0.95)
os.makedirs("visualizations", exist_ok=True)
SEED = 42

report = io.StringIO()


def log(*args):
    print(*args)
    print(*args, file=report)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"visualizations/{name}", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# DATA: binary skill matrix
# ---------------------------------------------------------------
df = pd.read_csv("jobs_clean_final.csv")
df = df[df["Has_Skills"]].reset_index(drop=True)
df["Skill_List"] = df["Required_Skills"].str.split(", ")
SKILLS = sorted({s for lst in df["Skill_List"] for s in lst})
X_all = pd.DataFrame({s: df["Skill_List"].map(lambda lst, s=s: int(s in lst)) for s in SKILLS})
groups_all = (df["Job_Title"].str.lower().str.strip() + "||" + df["Company"].str.lower().str.strip())

MAIN_ROLES = ["Software Engineering", "Frontend / Web Development", "Full-Stack Development", "Backend Development"]
LEVELS = ["Junior (0-2 yrs)", "Mid (3-4 yrs)", "Senior (5+ yrs)"]
TASKS = {
    "A. Job role": (df["Job_Role"].isin(MAIN_ROLES), "Job_Role", MAIN_ROLES),
    "B. Experience level": (df["Experience_Group"].isin(LEVELS), "Experience_Group", LEVELS),
}

MODELS = {
    "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                            random_state=SEED, n_jobs=-1),
}
SCORING = {"accuracy": "accuracy", "balanced_acc": "balanced_accuracy", "macro_f1": "f1_macro"}

results = []
oof_store = {}
for task, (mask, target, classes) in TASKS.items():
    X, y, g = X_all[mask].values, df.loc[mask, target].values, groups_all[mask].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    log("=" * 68)
    log(f"TASK {task}   (n = {len(y)} jobs, {len(np.unique(g))} distinct (title, company) groups)")
    log("=" * 68)
    log("Class balance:", dict(pd.Series(y).value_counts()))
    log(f"Majority-class accuracy: {pd.Series(y).value_counts(normalize=True).max():.3f}\n")

    for name, model in MODELS.items():
        cvres = cross_validate(model, X, y, groups=g, cv=cv, scoring=SCORING)
        row = {"Task": task, "Model": name}
        for k in SCORING:
            row[f"{k}_mean"] = cvres[f"test_{k}"].mean()
            row[f"{k}_std"] = cvres[f"test_{k}"].std()
        results.append(row)
        log(f"{name:28s} accuracy {row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f} | "
            f"balanced acc {row['balanced_acc_mean']:.3f} | macro-F1 {row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f}")

    # permutation test on the best model by macro-F1
    sub = pd.DataFrame([r for r in results if r["Task"] == task]).query("Model != 'Baseline (majority class)'")
    best_name = sub.sort_values("macro_f1_mean", ascending=False).iloc[0]["Model"]
    best = MODELS[best_name]
    # Permutation test: shuffle the labels over all jobs (null hypothesis = skills carry no
    # information about the target) and redo the same grouped cross-validation.
    # (sklearn's permutation_test_score shuffles *within* groups, which does nothing here
    # because most groups contain a single job, so the loop is written by hand.)
    perm_model = clone(best).set_params(n_estimators=40) if best_name == "Random Forest" else clone(best)
    N_PERM = 40
    rng = np.random.RandomState(SEED)
    score = cross_validate(perm_model, X, y, groups=g, cv=cv, scoring="f1_macro")["test_score"].mean()
    perm_scores = np.array([
        cross_validate(perm_model, X, rng.permutation(y), groups=g, cv=cv, scoring="f1_macro")["test_score"].mean()
        for _ in range(N_PERM)])
    pval = (np.sum(perm_scores >= score) + 1) / (N_PERM + 1)
    log(f"\nBest model: {best_name}. Permutation test ({N_PERM} label shuffles): "
        f"macro-F1 = {score:.3f} vs chance {perm_scores.mean():.3f} (95th pct {np.percentile(perm_scores, 95):.3f}), p = {pval:.3f}")

    # out-of-fold predictions -> confusion matrix and per-class report
    pred = cross_val_predict(best, X, y, groups=g, cv=cv)
    log("\nPer-class report (out-of-fold predictions):")
    log(classification_report(y, pred, labels=classes, digits=3, zero_division=0))
    oof_store[task] = (y, pred, classes, best_name)

res = pd.DataFrame(results)
res.round(4).to_csv("ml_model_comparison.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------
# 1. model comparison (macro-F1 and accuracy)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
for ax, (task, grp) in zip(axes, res.groupby("Task")):
    x = np.arange(len(grp))
    ax.bar(x - 0.2, grp["accuracy_mean"], 0.4, yerr=grp["accuracy_std"], label="Accuracy", color="#457b9d", capsize=3)
    ax.bar(x + 0.2, grp["macro_f1_mean"], 0.4, yerr=grp["macro_f1_std"], label="Macro-F1", color="#e76f51", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (majority class)", "\n(majority)").replace(" ", "\n", 1) for m in grp["Model"]], fontsize=8)
    ax.set(title=task, ylim=(0, 1))
    ax.legend(fontsize=8)
fig.suptitle("Model comparison (5-fold cross-validation, groups kept together)", y=1.02)
save(fig, "ml_model_comparison.png")

# 2. confusion matrices
for task, fname in [("A. Job role", "ml_confusion_role.png"), ("B. Experience level", "ml_confusion_experience.png")]:
    y, pred, classes, best_name = oof_store[task]
    cm = confusion_matrix(y, pred, labels=classes)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    short = [c.replace(" (0-2 yrs)", "").replace(" (3-4 yrs)", "").replace(" (5+ yrs)", "")
              .replace(" / Web Development", "").replace(" Development", "") for c in classes]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(cmn, annot=[[f"{cmn[i, j]:.0%}\n({cm[i, j]})" for j in range(len(classes))] for i in range(len(classes))],
                fmt="", cmap="Blues", vmin=0, vmax=1, xticklabels=short, yticklabels=short,
                cbar_kws={"label": "% of the true class"}, ax=ax)
    ax.set(title=f"{task}: {best_name}\n(out-of-fold predictions)", xlabel="Predicted", ylabel="True")
    save(fig, fname)

# 3. Task A: which skills characterise each role (logistic regression coefficients)
mask, target, classes = TASKS["A. Job role"]
Xr, yr = X_all[mask].values, df.loc[mask, target].values
lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Xr, yr)
coef = pd.DataFrame(lr.coef_, index=lr.classes_, columns=SKILLS).T
coef["Jobs_with_skill"] = Xr.sum(axis=0)
coef.round(3).to_csv("ml_role_coefficients.csv", encoding="utf-8-sig")
# very rare skills (< 40 jobs) get large but unstable coefficients: leave them out of the chart
MIN_JOBS = 40
coef = coef[coef["Jobs_with_skill"] >= MIN_JOBS].drop(columns="Jobs_with_skill")
log("=" * 68)
log(f"TASK A - skills that push a job towards each role (logistic regression; skills in >= {MIN_JOBS} jobs only)")
log("=" * 68)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, r in zip(axes.ravel(), MAIN_ROLES):
    top = coef[r].sort_values(ascending=False).head(6).iloc[::-1]
    ax.barh(top.index, top.values, color="#2a9d8f")
    ax.set_title(r, fontsize=10)
    ax.axvline(0, color="grey", lw=0.8)
    log(f"{r}: " + ", ".join(f"{s} ({v:+.2f})" for s, v in top.iloc[::-1].items()))
fig.suptitle("Skills most associated with each role (logistic regression coefficients)", y=1.0)
save(fig, "ml_role_top_skills.png")

with open("ml_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report.getvalue())
print("\nSaved: ml_model_comparison.csv, ml_role_coefficients.csv, ml_classification_report.txt, 4 charts ml_*.png")
