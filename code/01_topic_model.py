# === 2a. TOPIC MODEL on the items that contain harm clauses (BERTopic) =================
# Takes 5-15 minutes on the free GPU (Runtime > Change runtime type > T4 GPU), longer on CPU.
!pip install -q bertopic sentence-transformers
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

MAX_TOPICS = 25                                    # merge down to at most this many topics

d = load_items()
cl = pd.read_parquet(CLAUSES)
rows = np.sort(cl["row"].unique())
docs = d.loc[rows, "text"].fillna("").str.slice(0, 2000).tolist()
print(f"{len(docs):,} items with at least one harm clause")

emb = SentenceTransformer("all-MiniLM-L6-v2").encode(docs, batch_size=64, show_progress_bar=True)
vec = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))
tm = BERTopic(embedding_model=None, vectorizer_model=vec, min_topic_size=20,
              umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42))
topics, _ = tm.fit_transform(docs, emb)
if len(set(topics) - {-1}) > MAX_TOPICS:
    tm.reduce_topics(docs, nr_topics=MAX_TOPICS + 1)      # +1 because -1 (outliers) counts as a topic
    topics = tm.topics_
topics = tm.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb)   # assign outliers
tm.update_topics(docs, topics=topics, vectorizer_model=vec)

item_topics = pd.DataFrame({"row": rows, "topic": topics})
item_topics.to_parquet(f"{OUT}/item_topics.parquet", index=False)

# Template for labelling: one line per topic, to be filled in by hand
info = tm.get_topic_info()
info = info[info["Topic"] != -1]
tmpl = pd.DataFrame({
    "topic": info["Topic"],
    "n_items": info["Count"],
    "top_words": [", ".join(w for w, _ in tm.get_topic(t)[:10]) for t in info["Topic"]],
    "example_1": [ (tm.get_representative_docs(t) + ["", "", ""])[0][:300] for t in info["Topic"]],
    "example_2": [ (tm.get_representative_docs(t) + ["", "", ""])[1][:300] for t in info["Topic"]],
    "example_3": [ (tm.get_representative_docs(t) + ["", "", ""])[2][:300] for t in info["Topic"]],
    "label": "", "shelby_category": "", "keep": "",
})
tmpl.to_csv(f"{OUT}/topic_labels_TO_FILL.csv", index=False)
pd.set_option("display.max_colwidth", 120)
print(tmpl[["topic", "n_items", "top_words"]].to_string(index=False))
print(f"\nNow download {OUT}/topic_labels_TO_FILL.csv and fill three columns:")
print("  label            short name you give the topic (e.g. 'facial recognition and policing')")
print("  shelby_category  one of: representational, allocative, quality_of_service, interpersonal,")
print("                   social_system  (Shelby et al. 2023), or 'none' if not a harm topic")
print("  keep             1 if the topic is about an AI-related harm, 0 otherwise")
print("Save as topic_labels.csv, upload it, set LABELS below and run 2b.")

# === 2b. AGENT SHARES WITH THE TOPIC MIX HELD FIXED (after labelling) ==================
LABELS = "/content/topic_labels.csv"               # <- your filled-in file (Copy path)

with open(LABELS, encoding="utf-8-sig", errors="replace") as f:
    sep = max(["\t", ";", ","], key=f.readline().count)
lab = pd.read_csv(LABELS, sep=sep, encoding="utf-8-sig", encoding_errors="replace")
lab.columns = lab.columns.str.strip()
lab["keep"] = pd.to_numeric(lab["keep"], errors="coerce").fillna(0).astype(int)

cl = pd.read_parquet(CLAUSES)
it = pd.read_parquet(f"{OUT}/item_topics.parquet")
ct = cl.merge(it, on="row").merge(lab[["topic", "label", "shelby_category", "keep"]], on="topic", how="left")
kept = ct[ct["keep"] == 1].copy()
print(f"Clauses in kept harm topics: {len(kept):,} of {len(ct):,} ({len(kept)/len(ct)*100:.0f}%)")

for level in ["label", "shelby_category"]:
    print(f"\n=== Clauses per {level}, before / after ===")
    print(pd.crosstab(kept[level], kept["chatgpt"])[["before", "after"]].to_string())

# Topics with at least 20 clauses in both periods enter the fixed mix
for level in ["label", "shelby_category"]:
    ok = [g for g, x in kept.groupby(level)
          if (x["chatgpt"] == "before").sum() >= 20 and (x["chatgpt"] == "after").sum() >= 20]
    print(f"\n######## Topic level: {level}  ({len(ok)} groups with 20+ clauses in both periods)")
    for name, sub in {"all verbs": kept, "without replace": kept[kept["verb"] != "replace"]}.items():
        s = sub[sub[level].isin(ok)]
        res = pd.concat({"raw": boot_shares(s, level), "topic_mix_fixed": boot_shares(s, level, fixed_mix=True)}, axis=1)
        print(f"\n=== {name}: % with 95% CI ===")
        print(res.to_string())
        res.to_csv(f"{OUT}/topic_fixed_{level}_{name.replace(' ', '_')}.csv")

print("\n=== Within each topic (all verbs), % ===")
rows_ = []
for g, x in kept.groupby("label"):
    b, a = x[x["chatgpt"] == "before"], x[x["chatgpt"] == "after"]
    rows_.append({"topic": g, "shelby": x["shelby_category"].iloc[0], "n_before": len(b), "n_after": len(a),
                  "ai_before": round((b["grp"] == "ai_system").mean() * 100, 1) if len(b) else np.nan,
                  "ai_after": round((a["grp"] == "ai_system").mean() * 100, 1) if len(a) else np.nan,
                  "org_before": round((b["grp"] == "organisations").mean() * 100, 1) if len(b) else np.nan,
                  "org_after": round((a["grp"] == "organisations").mean() * 100, 1) if len(a) else np.nan})
wt = pd.DataFrame(rows_).sort_values("n_after", ascending=False)
print(wt.to_string(index=False))
wt.to_csv(f"{OUT}/within_topic.csv", index=False)

# === TOPIC TABLE FOR APPENDIX B: prints LaTeX rows ======================================
NAMES = {0: ("Job displacement", "Social system"), 1: ("Facial recognition and policing", "Allocative"),
         2: ("Personal data", "Interpersonal"), 3: ("Social media", "Social system"),
         4: ("OpenAI and the AI industry", "Interpersonal"), 5: ("China and national security", "Social system"),
         6: ("Government surveillance", "Interpersonal"), 7: ("Buying, selling and theft", "--"),
         8: ("Workers fired", "Allocative"), 9: ("Identity theft", "Allocative"),
         10: ("Health care", "Quality of service"), 11: ("Google", "Interpersonal")}
cl = pd.read_parquet(CLAUSES).merge(pd.read_parquet(TOPICS), on="row")
def pct(g, cat):
    return f"{(g['grp'] == cat).mean()*100:.1f}" if len(g) else "--"
for t, (name, cat) in NAMES.items():
    g = cl[cl["topic"] == t]
    b, a = g[g["chatgpt"] == "before"], g[g["chatgpt"] == "after"]
    print(f"{t} & {name} & {cat} & {len(b):,} / {len(a):,} & {pct(b,'ai_system')} / {pct(a,'ai_system')} & "
          f"{pct(b,'organisations')} / {pct(a,'organisations')} \\\\")
print(f"\nTotal clauses: {len(cl):,}  (check: should be 2,704)")
