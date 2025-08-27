# Run iter-DBSCAN
# Tuning cheat-sheet

# Start with initial_distance ∈ [0.25, 0.35] for MiniLM; a bit lower for MPNet (0.22–0.30).

# initial_minimum_samples: 12–32 (depends on dataset size; larger sets → larger start).

# If too much noise (-1): raise max_iteration, raise delta_distance, or lower initial_minimum_samples.

# If giant generic cluster: lower threshold (e.g., 100–200) and/or reduce max_iteration.

# ⚠️ Memory note: the precomputed distance matrix is n×n. If n is large (e.g., >30k), consider running per domain/topic or use mini-batching/approx neighbors.

clust = IterDBSCAN(
    initial_distance=0.30,      # start tighter
    initial_minimum_samples=16, # demand denser cores first
    delta_distance=0.01,        # relax eps each round
    delta_minimum_samples=1,    # and ease min_samples
    max_iteration=15,
    threshold=300               # ignore too-big clusters (likely generic)
)
labels = clust.fit_predict(emb)
df["cluster"] = labels

# Inspect and save results

# Save a flat file for review
df[["conv_id","speaker","text","cluster"]].to_csv("clustered_conversations.csv", index=False)

# Quick cluster counts
print(df["cluster"].value_counts().sort_index())

# Auto-label clusters (intent names) with class-TF-IDF

# This gives you top words per cluster so you can label intents quickly.

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

docs = df[df["cluster"]!=-1].copy()
docs["cluster"] = docs["cluster"].astype(int)

# Build one "document" per cluster by concatenating its utterances
grouped = docs.groupby("cluster")["text_clean"].apply(lambda x: " ".join(x)).reset_index()

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.6)
Xc = vectorizer.fit_transform(grouped["text_clean"])
vocab = np.array(vectorizer.get_feature_names_out())

def top_terms(row, k=8):
    idx = row.toarray().ravel().argsort()[::-1][:k]
    return ", ".join(vocab[idx])

grouped["top_terms"] = [top_terms(Xc[i]) for i in range(Xc.shape[0])]
grouped.head(10)

# Evaluate quickly (optional)

# Silhouette on non-noise points:

from sklearn.metrics import silhouette_score

mask = df["cluster"]!=-1
if mask.sum() > 1 and len(df["cluster"][mask].unique()) > 1:
    print("Silhouette:", silhouette_score(emb[mask.values], df["cluster"][mask].values, metric="cosine"))

# Curate & iterate

# Merge or split clusters after quick human review (tiny clusters may be true long-tail intents; generic clusters may need re-runs with stricter params).

# Keep a lookup table {cluster_id → intent_name} and a few canonical examples per intent for QA.
