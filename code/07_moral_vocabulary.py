# === 7. MORAL VOCABULARY: corpus trend, by agent of harm, topic mix fixed ================
import os, re
import numpy as np
import pandas as pd

CLEAN   = "/content/old_clean.parquet"
CLAUSES = "/content/harm_clauses_v3.parquet"
MFD_DIC = "/content/mft_original.dic"
TOPICS  = "/content/item_topics.parquet"
LABELS  = "/content/topic_labels.csv"
OUT     = "/content/extra2"
os.makedirs(OUT, exist_ok=True)
for p in [CLEAN, CLAUSES, MFD_DIC, TOPICS, LABELS]:
    print(("OK      " if os.path.exists(p) else "MISSING ") + p)

PERIODS = ["2020-2021", "2022-2023", "2024-2025"]
CHATGPT = pd.Timestamp("2022-11-30", tz="UTC")

def load_items():
    d = pd.read_parquet(CLEAN)
    d = d[d["period"].isin(PERIODS)].reset_index(drop=True)
    dt = pd.to_datetime(d["created_utc"], unit="s", utc=True)
    d["year"] = dt.dt.year
    d["chatgpt"] = np.where(dt >= CHATGPT, "after", "before")
    return d

def boot_ci(values, point, decimals=1):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"{point:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"

TECH_SENSES = [r"submission statement", r"\bdiscriminators?\b", r"\bdecentrali[sz]\w*", r"\bhonestly\b"]
MAIN = ["care.vice", "fairness.vice"]

def read_dic(path):
    cats, words, part = {}, {}, 0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("%"):
            part += 1; continue
        f = line.split()
        if part == 1:
            cats[f[0]] = f[1]
        elif part >= 2:
            words[f[0].lower()] = [cats[c] for c in f[1:] if c in cats]
    return words
words = read_dic(MFD_DIC)
pats = [re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w)
        for w, cs in words.items() if set(cs) & set(MAIN)]
rx = re.compile(r"\b(?:" + "|".join(sorted(pats, key=len, reverse=True)) + r")\b")
print(f"{len(pats)} dictionary entries in {MAIN}")

def clean(s):
    s = s.fillna("").str.lower()
    for t in TECH_SENSES:
        s = s.str.replace(t, " ", regex=True)
    return s
rng = np.random.default_rng(5)
def rate(h, n, B=2000):
    h, n = np.asarray(h, float), np.asarray(n, float)
    idx = rng.integers(0, len(h), (B, len(h)))
    return boot_ci(h[idx].sum(1) / n[idx].sum(1) * 1000, h.sum() / n.sum() * 1000, 2)

# --- A. Corpus: per 1,000 words, raw and with the subreddit mix fixed ---
d = load_items()
txt = clean(d["text"])
d["n_words"], d["n_hits"] = txt.str.count(r"[a-z']+"), txt.str.count(rx)
d = d[d["n_words"] > 0]
out = {}
for label, sub in [("posts", d[d["content_type"] == "post"]), ("comments", d[d["content_type"] == "comment"])]:
    r = {str(y): rate(g["n_hits"], g["n_words"]) for y, g in sub.groupby("year")}
    w = sub["subreddit"].value_counts(normalize=True)
    for g in ["before", "after"]:
        part = sub[sub["chatgpt"] == g]
        r[g] = rate(part["n_hits"], part["n_words"])
        by = part.groupby("subreddit")[["n_hits", "n_words"]].sum()
        r[g + " (subreddit mix fixed)"] = f"{(by['n_hits'] / by['n_words'] * 1000 * w.reindex(by.index)).sum() / w.reindex(by.index).sum():.2f}"
    out[label] = r
print("\n=== A. care.vice + fairness.vice per 1,000 words, 95% CI ===")
print(pd.DataFrame(out).to_string())

# --- B. Harm sentences: is moral vocabulary rarer when the AI system is the agent? ---
# (the sentence's own harm verb is not counted)
cl = pd.read_parquet(CLAUSES)
# remove the clause's own harm verb (and its derivatives), since many harm verbs are themselves MFD words
s = clean(pd.Series([re.sub(r"\b" + re.escape(re.sub(r"(e|y)$", "", str(v))) + r"\w*", " ", str(t).lower())
                     for t, v in zip(cl["sentence"], cl["verb"])], index=cl.index))
cl["n_words"], cl["n_hits"] = s.str.count(r"[a-z']+"), s.str.count(rx)
cl["agent3"] = np.where(cl["grp"].isin(["ai_system", "organisations"]), cl["grp"],
                np.where(cl["agent_category"].astype(str).str.startswith("no_agent"), "agentless passive", "other"))
res = {}
for name, sub in {"all verbs": cl, "without replace": cl[cl["verb"] != "replace"]}.items():
    for a in ["ai_system", "organisations", "agentless passive"]:
        g = sub[sub["agent3"] == a]
        res[(name, a)] = {"clauses": len(g),
                          "per 1,000 words": rate(g["n_hits"], g["n_words"]),
                          "% sentences with any": f"{(g['n_hits'] > 0).mean()*100:.1f}"}
print("\n=== B. Moral vocabulary in harm sentences, by agent ===")
print(pd.DataFrame(res).T.to_string())
pd.DataFrame(out).to_csv(f"{OUT}/mfd_corpus.csv"); pd.DataFrame(res).T.to_csv(f"{OUT}/mfd_by_agent.csv")

# --- C. Items with harm clauses: moral vocabulary with the topic mix fixed ---
it = pd.read_parquet(TOPICS).merge(pd.read_csv(LABELS)[["topic", "label", "keep"]], on="topic")
it = it[it["keep"] == 1].join(d[["n_hits", "n_words", "chatgpt", "content_type"]], on="row", how="inner")
resC = {}
for label, sub in [("posts and comments", it), ("comments", it[it["content_type"] == "comment"])]:
    w = sub.groupby("label")["n_words"].sum(); w = w / w.sum()
    for g in ["before", "after"]:
        by = sub[sub["chatgpt"] == g].groupby("label")[["n_hits", "n_words"]].sum()
        resC[(label, g)] = {"raw": round(by["n_hits"].sum() / by["n_words"].sum() * 1000, 2),
                            "topic mix fixed": round((by["n_hits"] / by["n_words"] * 1000 * w.reindex(by.index)).sum()
                                                     / w.reindex(by.index).sum(), 2)}
print("\n=== C. Items with harm clauses: care.vice + fairness.vice per 1,000 words ===")
print(pd.DataFrame(resC).T.to_string())
pd.DataFrame(resC).T.to_csv(f"{OUT}/mfd_topicfixed.csv")

# === 7b. MORAL VOCABULARY: CIs for the topic-fixed estimate, and item-level link to agents ===
# Run after cell 7 (uses d, cl, it, rate, OUT).
rng = np.random.default_rng(11)

def topic_fixed(sub, B=2000):
    """Topic-fixed rate per 1,000 words before and after, with item bootstrap CIs."""
    labs = sorted(sub["label"].unique()); T = len(labs)
    code = (sub["chatgpt"] == "after").to_numpy().astype(int) * T + sub["label"].map({l: i for i, l in enumerate(labs)}).to_numpy()
    N = len(sub)
    idx = np.vstack([np.arange(N), rng.integers(0, N, (B, N))])
    flat = (np.arange(B + 1)[:, None] * 2 * T + code[idx]).ravel()
    def sums(v):
        return np.bincount(flat, weights=np.asarray(v, float)[idx].ravel(), minlength=(B + 1) * 2 * T).reshape(B + 1, 2, T)
    h, n = sums(sub["n_hits"]), sums(sub["n_words"])
    w = n.sum(1) / n.sum((1, 2))[:, None]                      # pooled word share of each topic
    rate_t = np.where(n > 0, h / np.maximum(n, 1), np.nan) * 1000
    est = {g: np.nansum(rate_t[:, i] * w, 1) / np.nansum(np.where(np.isnan(rate_t[:, i]), 0, w), 1) for i, g in enumerate(["before", "after"])}
    diff = est["after"] - est["before"]
    return {"before": boot_ci(est["before"][1:], est["before"][0], 2), "after": boot_ci(est["after"][1:], est["after"][0], 2),
            "change": boot_ci(diff[1:], diff[0], 2)}

resD = {lab: topic_fixed(sub) for lab, sub in [("posts and comments", it), ("posts", it[it["content_type"] == "post"]),
                                               ("comments", it[it["content_type"] == "comment"])]}
print("=== D1. care.vice + fairness.vice per 1,000 words, topic mix fixed, 95% CI ===")
print(pd.DataFrame(resD).T.to_string())

# Item level: whole texts of items whose harm clauses name only AI, only organisations, or only agentless passives
g = cl.groupby("row")["agent3"].agg(lambda x: set(x) & {"ai_system", "organisations", "agentless passive"})
only = g[g.apply(len) == 1].apply(lambda s: next(iter(s)))
items = d.loc[d.index.intersection(only.index)].assign(agent=only)
resE = {}
for typ, sub in [("posts and comments", items), ("comments", items[items["content_type"] == "comment"])]:
    for a in ["ai_system", "organisations", "agentless passive"]:
        x = sub[sub["agent"] == a]
        resE[(typ, a)] = {"items": len(x), "per 1,000 words": rate(x["n_hits"], x["n_words"])}
print("\n=== D2. Moral vocabulary in whole items, by the agent their harm clauses name ===")
print(pd.DataFrame(resE).T.to_string())
pd.DataFrame(resD).T.to_csv(f"{OUT}/mfd_topicfixed_ci.csv"); pd.DataFrame(resE).T.to_csv(f"{OUT}/mfd_items_by_agent.csv")
