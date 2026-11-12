import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=" * 65)
print("       K-MEANS CLUSTERING AND CLUSTER INTERPRETATION")
print("=" * 65)

# =====================================================
# 1. LOAD DATASET (cleaned file produced by 01_cleaning.py)
# =====================================================

df = pd.read_csv("jobs_clean_final.csv")
df = df[df["Has_Skills"]]

print("\nJobs with skills:", len(df))

# =====================================================
# 2. PREPARE ONE DOCUMENT PER JOB
# =====================================================

jobs = df[["Job_Number", "Job_Title", "Required_Skills"]].rename(
    columns={"Required_Skills": "Skills"}
).reset_index(drop=True)

jobs["Text"] = jobs["Job_Title"] + " " + jobs["Skills"]

print("Number of jobs:", len(jobs))

# =====================================================
# 3. TF-IDF VECTORIZATION
# =====================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X = vectorizer.fit_transform(jobs["Text"])

print("TF-IDF matrix:", X.shape)

# =====================================================
# 4. COMPARE DIFFERENT K VALUES
# =====================================================

results = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)

    results.append({
        "K": k,
        "Silhouette_Score": score,
        "Inertia": model.inertia_
    })

    print(
        f"K={k} | "
        f"Silhouette Score={score:.4f}"
    )

results_df = pd.DataFrame(results)

# =====================================================
# 5. SELECT BEST K
# =====================================================

best_k = int(
    results_df.loc[
        results_df["Silhouette_Score"].idxmax(),
        "K"
    ]
)

print("\nBest K:", best_k)

# =====================================================
# 6. FINAL K-MEANS MODEL
# =====================================================

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

jobs["Cluster"] = final_model.fit_predict(X)

# =====================================================
# 7. CLUSTER DISTRIBUTION
# =====================================================

summary = jobs.groupby("Cluster").agg(
    Number_of_Jobs=("Job_Number", "count")
).reset_index()

print("\nCLUSTER DISTRIBUTION")
print(summary.to_string(index=False))

# =====================================================
# 8. CLUSTER INTERPRETATION
# =====================================================

print("\n" + "=" * 65)
print("             CLUSTER INTERPRETATION")
print("=" * 65)

terms = vectorizer.get_feature_names_out()

cluster_details = []

for cluster_id in range(best_k):

    center = final_model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[-10:][::-1]

    top_terms = [terms[i] for i in top_indices]

    cluster_jobs = jobs[
        jobs["Cluster"] == cluster_id
    ]

    print("\n" + "-" * 60)
    print(f"CLUSTER {cluster_id}")
    print("-" * 60)

    print("Number of jobs:", len(cluster_jobs))

    print("\nTop 10 keywords:")
    print(", ".join(top_terms))

    print("\nSample job titles:")

    for title in cluster_jobs["Job_Title"].head(10):
        print("-", title)

    cluster_details.append({
        "Cluster": cluster_id,
        "Number_of_Jobs": len(cluster_jobs),
        "Top_Keywords": ", ".join(top_terms),
        "Sample_Jobs": " | ".join(
            cluster_jobs["Job_Title"].head(10)
        )
    })

# =====================================================
# 9. SAVE RESULTS
# =====================================================

jobs.to_csv(
    "kmeans_job_clusters.csv",
    index=False,
    encoding="utf-8-sig"
)

results_df.to_csv(
    "kmeans_evaluation.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    "kmeans_cluster_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(cluster_details).to_csv(
    "kmeans_cluster_interpretation.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 10. VISUALIZE SILHOUETTE SCORES
# =====================================================

plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["Silhouette_Score"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means - Silhouette Score Comparison")

plt.xticks(range(2, 11))
plt.grid(True)

plt.savefig(
    "kmeans_silhouette_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 11. VISUALIZE CLUSTER DISTRIBUTION
# =====================================================

plt.figure(figsize=(10, 5))

plt.bar(
    summary["Cluster"].astype(str),
    summary["Number_of_Jobs"]
)

plt.xlabel("Cluster")
plt.ylabel("Number of Jobs")
plt.title("Distribution of Jobs Across Clusters")

plt.savefig(
    "kmeans_cluster_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 12. FINAL MESSAGE
# =====================================================

print("\n" + "=" * 65)
print("K-MEANS ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated files:")
print("- kmeans_job_clusters.csv")
print("- kmeans_evaluation.csv")
print("- kmeans_cluster_summary.csv")
print("- kmeans_cluster_interpretation.csv")
print("- kmeans_silhouette_comparison.png")
print("- kmeans_cluster_distribution.png")