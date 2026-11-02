# === 8a. VALIDATION SAMPLE FOR THE DICTIONARY: 150 matches to code by hand ===============
# Run after cell 7 (uses d, rx, clean, OUT). 75 matches before and 75 after ChatGPT.
rng = np.random.default_rng(21)
txt = clean(d["text"])
rows_ = []
for i, t in txt.items():
    for m in rx.finditer(t):
        rows_.append((i, m.group(0), m.start(), m.end()))
hits = pd.DataFrame(rows_, columns=["row", "word", "start", "end"])
hits["chatgpt"] = d.loc[hits["row"], "chatgpt"].to_numpy()
hits["content_type"] = d.loc[hits["row"], "content_type"].to_numpy()
samp = hits.groupby("chatgpt", group_keys=False).sample(75, random_state=22).sample(frac=1, random_state=23).reset_index(drop=True)
def context(r, n=150):
    t = txt[r["row"]]
    return (t[max(0, r["start"] - n):r["start"]] + " >>" + t[r["start"]:r["end"]].upper() + "<< " + t[r["end"]:r["end"] + n]).replace("\n", " ")
samp["id"] = np.arange(len(samp))
samp["context"] = samp.apply(context, axis=1)
samp[["id", "row", "word", "chatgpt", "content_type"]].to_csv(f"{OUT}/mfd_key.csv", index=False)
samp[["id", "word", "context"]].assign(moral="").to_csv(f"{OUT}/mfd_150_TO_CODE.csv", index=False)
print(f"{len(hits):,} matches in total. Written {OUT}/mfd_150_TO_CODE.csv")
print("""
Code each row:  moral = 1 if the marked word is used in a moral sense (harm, cruelty, suffering,
unfairness, prejudice, discrimination done to someone); 0 if technical, statistical, figurative or
otherwise non-moral (e.g. 'bias' of an estimator, 'console wars', 'kill the process', 'damage' to a
device). Code the marked word only, not the whole sentence. Save as mfd_150.csv and run 8a-b.""")

# === MFD validation: read coded Excel, recode, precision before/after ======================
import os
import numpy as np
import pandas as pd

CODED = "/content/mfd_150_TO_CODE (2).xlsx"     # your coded file
KEY   = "/content/extra2/mfd_key.csv"           # written by cell 8a (which period each id is from)
RECODE_TO_1 = [2, 7, 15, 30, 34, 35, 65, 74, 77, 81, 95, 149, 111]   # remove 111 if you keep it as 0

for p in [CODED, KEY]:
    print(("OK      " if os.path.exists(p) else "MISSING ") + p)

v = pd.read_excel(CODED)
v.columns = v.columns.str.strip().str.lower()
v["moral"] = pd.to_numeric(v["moral"], errors="coerce")
print(f"\nRows: {len(v)}, coded: {v['moral'].notna().sum()}, moral = 1 before recoding: {int(v['moral'].sum())}")

v.loc[v["id"].isin(RECODE_TO_1), "moral"] = 1
v[["id", "word", "context", "moral"]].to_csv("/content/mfd_150.csv", index=False)
print(f"After recoding: moral = 1 in {int(v['moral'].sum())} of {len(v)}  (saved as /content/mfd_150.csv)")

v = v.merge(pd.read_csv(KEY)[["id", "chatgpt"]], on="id")
rng = np.random.default_rng(24)
print("\n=== Precision of care.vice + fairness.vice matches, 95% bootstrap CI ===")
for g in ["before", "after", "all"]:
    x = (v if g == "all" else v[v["chatgpt"] == g])["moral"].dropna().to_numpy()
    b = rng.choice(x, (2000, len(x))).mean(1) * 100
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f"{g:7s} n = {len(x):3d}  precision = {x.mean()*100:.1f} [{lo:.1f}, {hi:.1f}]%  ({int(x.sum())} moral)")

print("\n=== By word (words with at least 3 matches) ===")
w = v.groupby(v["word"].str.lower())["moral"].agg(n="size", moral="sum")
w["precision_%"] = (w["moral"] / w["n"] * 100).round(0)
print(w[w["n"] >= 3].sort_values("n", ascending=False).to_string())

# === 8a-b. PRECISION OF THE DICTIONARY, BEFORE AND AFTER ==================================
MFD_CODED = "/content/mfd_150.csv"                      # <- your coded file
v = pd.read_csv(MFD_CODED, sep=None, engine="python").merge(pd.read_csv(f"{OUT}/mfd_key.csv")[["id", "chatgpt"]], on="id")
v["moral"] = pd.to_numeric(v["moral"], errors="coerce")
rng = np.random.default_rng(24)
for g in ["before", "after", "all"]:
    x = (v if g == "all" else v[v["chatgpt"] == g])["moral"].dropna().to_numpy()
    b = rng.choice(x, (2000, len(x))).mean(1) * 100
    print(f"{g:7s} n = {len(x):3d}  precision = {boot_ci(b, x.mean() * 100)}%")
print("\nNon-moral matches:")
print(v[v["moral"] == 0][["id", "word", "chatgpt"]].to_string(index=False))

# === 8b. MORAL VOCABULARY: which words, same authors, and agent with topic held constant ===
# Run after cells 7 and 7b (uses d, cl, it, items, rx, words, MAIN, clean, rate, boot_ci, OUT).
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smf
com = d[d["content_type"] == "comment"]
ctxt = clean(com["text"])

# --- (1) Which dictionary words carry the decline in comments? (per 1,000 words) ---
nw = {g: com.loc[com["chatgpt"] == g, "n_words"].sum() for g in ["before", "after"]}
cnt = {g: Counter(m.group(0) for t in ctxt[com["chatgpt"] == g] for m in rx.finditer(t)) for g in ["before", "after"]}
wd = pd.DataFrame({g: pd.Series(cnt[g]) for g in cnt}).fillna(0)
wd["rate_before"], wd["rate_after"] = wd["before"] / nw["before"] * 1000, wd["after"] / nw["after"] * 1000
wd["change"] = wd["rate_after"] - wd["rate_before"]
wd["share_of_decline_%"] = wd["change"] / wd["change"].sum() * 100
print("=== (1) Dictionary words with the largest contribution to the change in comments ===")
print(wd.sort_values("change").head(15).round(3).to_string())
print(f"Top 5 words account for {wd.sort_values('change').head(5)['share_of_decline_%'].sum():.0f}% of the net change")

# --- (2) Words rising and falling overall in comments (log-odds, informative Dirichlet prior; Monroe et al. 2008) ---
tok = {g: Counter(w for t in ctxt[com["chatgpt"] == g] for w in re.findall(r"[a-z][a-z']+", t)) for g in ["before", "after"]}
V = pd.DataFrame(tok).fillna(0); V = V[V.sum(axis=1) >= 20]
a0 = 1000; prior = V.sum(axis=1) / V.sum(axis=1).sum() * a0
na, nb = V["after"].sum(), V["before"].sum()
delta = (np.log((V["after"] + prior) / (na + a0 - V["after"] - prior))
         - np.log((V["before"] + prior) / (nb + a0 - V["before"] - prior)))
z = delta / np.sqrt(1 / (V["after"] + prior) + 1 / (V["before"] + prior))
print("\n=== (2) Comments: words most over-represented AFTER ChatGPT (z) ===")
print(", ".join(f"{w} {s:.0f}" for w, s in z.sort_values(ascending=False).head(40).items()))
print("\n=== Comments: words most over-represented BEFORE ChatGPT (z) ===")
print(", ".join(f"{w} {-s:.0f}" for w, s in z.sort_values().head(40).items()))
z.rename("z").to_csv(f"{OUT}/mfd_logodds_comments.csv")

# --- (3) Same authors: comments by authors active in both periods ---
com = com.assign(author=com["author"].astype(str).str.strip())
com = com[~com["author"].isin(["", "None", "nan", "[deleted]", "[removed]"])]
per = com.groupby("author")["chatgpt"].nunique()
both = com[com["author"].isin(per[per == 2].index)]
rng = np.random.default_rng(31)
def author_rate(sub, B=2000):
    au = sub["author"].unique(); pos = {a: i for i, a in enumerate(au)}; ai = sub["author"].map(pos).to_numpy()
    out = {}
    for g in ["before", "after"]:
        m = (sub["chatgpt"] == g).to_numpy()
        h = np.bincount(ai, weights=sub["n_hits"].to_numpy() * m, minlength=len(au))
        n = np.bincount(ai, weights=sub["n_words"].to_numpy() * m, minlength=len(au))
        draw = rng.integers(0, len(au), (B, len(au)))
        out[g] = (h[draw].sum(1) / n[draw].sum(1) * 1000, h.sum() / n.sum() * 1000)
    diff = out["after"][0] - out["before"][0]
    return {"authors": len(au), "before": boot_ci(*out["before"], 2), "after": boot_ci(*out["after"], 2),
            "change": boot_ci(diff, out["after"][1] - out["before"][1], 2)}
print("\n=== (3) Comments by authors active in both periods, per 1,000 words, author-clustered CI ===")
print(pd.Series(author_rate(both)).to_string())

# --- (4) Agent of harm and moral vocabulary, holding topic and period constant ---
reg = items.join(it.set_index("row")[["label"]], how="inner")
reg = reg[reg["n_words"] > 0].assign(log_words=lambda x: np.log(x["n_words"]))
reg["agent"] = pd.Categorical(reg["agent"], categories=["organisations", "ai_system", "agentless passive"])
# Poisson rate model with robust (sandwich) standard errors: valid under overdispersion
m = smf.glm("n_hits ~ C(agent) + C(label) + C(chatgpt, Treatment('before')) + C(content_type)",
            data=reg, family=sm.families.Poisson(), offset=reg["log_words"]).fit(cov_type="HC1")
tab = pd.DataFrame({"IRR": np.exp(m.params), "lo": np.exp(m.conf_int()[0]), "hi": np.exp(m.conf_int()[1]), "p": m.pvalues})
print(f"\n=== (4) Poisson rate model (robust SE), n = {len(reg)} items: moral words ~ agent + topic + period + type ===")
print(tab.loc[[i for i in tab.index if i.startswith("C(agent)") or i.startswith("C(chatgpt")]].round(3).to_string())
print("IRR < 1 for ai_system = fewer moral words than items naming organisations, same topic and period.")
tab.to_csv(f"{OUT}/mfd_ratemodel.csv")
