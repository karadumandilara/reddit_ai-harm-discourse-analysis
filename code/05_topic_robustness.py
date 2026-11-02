# === 5. TOPIC MODEL ROBUSTNESS: seeds, topic sizes and number of topics ================
# Refits BERTopic with three other UMAP seeds and minimum topic sizes of 15 and 30, and
# with the number of topics fixed at 8 and 15, and repeats the topic-fixed estimate.
# All topics are kept for every model so the rows are comparable (Table A9).
# GPU recommended (Colab: Runtime > Change runtime type > T4 GPU).
!pip install -q bertopic sentence-transformers
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

CLEAN   = "/content/old_clean.parquet"
CLAUSES = "/content/harm_clauses_v3.parquet"
TOPICS  = "/content/item_topics.parquet"
OUT     = "/content/extra2"
os.makedirs(OUT, exist_ok=True)
for p in [CLEAN, CLAUSES, TOPICS]:
    print(("OK      " if os.path.exists(p) else "MISSING ") + p)

PERIODS = ["2020-2021", "2022-2023", "2024-2025"]
CATS = ("ai_system", "organisations")

d = pd.read_parquet(CLEAN)
d = d[d["period"].isin(PERIODS)].reset_index(drop=True)
cl = pd.read_parquet(CLAUSES)
rows = np.sort(cl["row"].unique())
docs = d.loc[rows, "text"].fillna("").str.slice(0, 2000).tolist()
emb = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, batch_size=64, show_progress_bar=True)


def fixed_mix(cl_t, cat, min_n=20):
    """Topic-fixed share before/after, topics with min_n clauses in both periods."""
    n = cl_t.groupby(["topic", "chatgpt"]).size().unstack(fill_value=0)
    ok = n[(n.get("before", 0) >= min_n) & (n.get("after", 0) >= min_n)].index
    s = cl_t[cl_t["topic"].isin(ok)]
    w = s["topic"].value_counts(normalize=True)
    out = {}
    for g in ["before", "after"]:
        p = s[s["chatgpt"] == g].groupby("topic")["grp"].apply(lambda x: (x == cat).mean())
        out[g] = float((p * w.reindex(p.index)).sum() / w.reindex(p.index).sum() * 100)
    return out, len(ok)


def evaluate(topics, label):
    cl_t = cl.merge(pd.DataFrame({"row": rows, "topic": topics}), on="row")
    res = {"model": label, "n_topics": len(set(topics))}
    for name, sub in {"all": cl_t, "no_replace": cl_t[cl_t["verb"] != "replace"]}.items():
        for cat in CATS:
            v, k = fixed_mix(sub, cat)
            res[f"{cat}_{name}"] = f"{v['before']:.1f} -> {v['after']:.1f}"
        res[f"topics_used_{name}"] = k
    return res


def fit(seed=42, mts=20, nr=None, max_topics=25):
    # min_df=1: the vectoriser only builds topic words; topic assignments come from the
    # embeddings, so this setting does not affect the estimates.
    vec = CountVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    tm = BERTopic(vectorizer_model=vec, min_topic_size=mts,
                  umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                                  metric="cosine", random_state=seed))
    t, _ = tm.fit_transform(docs, emb)
    n_found = len(set(t) - {-1})
    target = nr if nr is not None else (max_topics if n_found > max_topics else None)
    if target is not None and n_found > target:
        tm.reduce_topics(docs, nr_topics=target + 1)   # +1 because -1 (outliers) counts as a topic
        t = tm.topics_
    t = tm.reduce_outliers(docs, t, strategy="embeddings", embeddings=emb)
    print(f"  seed {seed}, min size {mts}, target {target}: {n_found} topics found, "
          f"{len(set(t))} after reduction")
    return np.array(t)


results = [evaluate(pd.read_parquet(TOPICS).set_index("row").loc[rows, "topic"].to_numpy(),
                    "main model (all topics)")]
for seed in [1, 2, 3]:
    for mts in [15, 30]:
        results.append(evaluate(fit(seed=seed, mts=mts), f"seed {seed}, min size {mts}"))
for nr in [8, 15]:
    results.append(evaluate(fit(seed=42, mts=10, nr=nr), f"fixed {nr} topics"))

res = pd.DataFrame(results)
print("\n=== Topic-fixed shares (before -> after), %, under alternative topic models ===")
print(res.to_string(index=False))
res.to_csv(f"{OUT}/topic_robustness.csv", index=False)
