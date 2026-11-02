# === 9. VALIDATED SUBSET: only dictionary words that were mostly moral in the coded sample ===
# Self-contained. Words kept: precision >= 70% in the 150 coded matches (harm, violence, abuse,
# discrimination, killed, harmful, violent, war). Dropped: bias*, fight*, attack*, destroy, killer, ruin*.
import os, re
import numpy as np
import pandas as pd

CLEAN  = "/content/old_clean.parquet"
TOPICS = "/content/item_topics.parquet"
LABELS = "/content/topic_labels.csv"
PERIODS = ["2020-2021", "2022-2023", "2024-2025"]
CHATGPT = pd.Timestamp("2022-11-30", tz="UTC")
SUBSET = re.compile(r"\b(?:harm(?:s|ed|ing|ful|fully)?|violen\w*|abus\w*|discriminat(?:e|es|ed|ing|ion|ions|ory)"
                    r"|kill(?:s|ed|ing)?|wars?)\b")
TECH = [r"submission statement", r"\bdiscriminators?\b", r"\bkill(?:s|ed|ing)? (?:the |a |an )?(?:process|app|feature|switch|battery|bill)\w*",
        r"\b(?:console|browser|price|format|bidding|streaming|talent|culture|flame) wars?\b"]

rng = np.random.default_rng(41)
def boot_ci(v, p, dec=2):
    lo, hi = np.percentile(v, [2.5, 97.5]); return f"{p:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"
def rate(h, n, B=2000):
    h, n = np.asarray(h, float), np.asarray(n, float)
    idx = rng.integers(0, len(h), (B, len(h)))
    return h[idx].sum(1) / n[idx].sum(1) * 1000, h.sum() / n.sum() * 1000

d = pd.read_parquet(CLEAN)
d = d[d["period"].isin(PERIODS)].reset_index(drop=True)
dt = pd.to_datetime(d["created_utc"], unit="s", utc=True)
d["year"], d["chatgpt"] = dt.dt.year, np.where(dt >= CHATGPT, "after", "before")
txt = d["text"].fillna("").str.lower()
for t in TECH:
    txt = txt.str.replace(t, " ", regex=True)
d["n_words"], d["n_hits"] = txt.str.count(r"[a-z']+"), txt.str.count(SUBSET)
d = d[d["n_words"] > 0]

print("=== Validated subset per 1,000 words, 95% CI ===")
for typ in ["comment", "post"]:
    sub = d[d["content_type"] == typ]
    yearly = ", ".join(f"{y}: {g['n_hits'].sum() / g['n_words'].sum() * 1000:.2f}" for y, g in sub.groupby("year"))
    b, a = [rate(sub.loc[sub["chatgpt"] == g, "n_hits"], sub.loc[sub["chatgpt"] == g, "n_words"]) for g in ["before", "after"]]
    w = sub["subreddit"].value_counts(normalize=True)
    fx = {}
    for g in ["before", "after"]:
        by = sub[sub["chatgpt"] == g].groupby("subreddit")[["n_hits", "n_words"]].sum()
        fx[g] = (by["n_hits"] / by["n_words"] * 1000 * w.reindex(by.index)).sum() / w.reindex(by.index).sum()
    print(f"\n{typ}s  yearly: {yearly}")
    print(f"  before {boot_ci(*b)}   after {boot_ci(*a)}   change {boot_ci(a[0] - b[0], a[1] - b[1])}")
    print(f"  subreddit mix fixed: {fx['before']:.2f} -> {fx['after']:.2f}")

# Topic mix fixed, items with harm clauses (comments)
it = pd.read_parquet(TOPICS).merge(pd.read_csv(LABELS)[["topic", "label", "keep"]], on="topic")
it = it[it["keep"] == 1].join(d[["n_hits", "n_words", "chatgpt", "content_type"]], on="row", how="inner")
for typ, sub in [("comments", it[it["content_type"] == "comment"]), ("posts and comments", it)]:
    labs = sorted(sub["label"].unique()); T = len(labs); N = len(sub); B = 2000
    code = (sub["chatgpt"] == "after").to_numpy().astype(int) * T + sub["label"].map({l: i for i, l in enumerate(labs)}).to_numpy()
    idx = np.vstack([np.arange(N), rng.integers(0, N, (B, N))])
    flat = (np.arange(B + 1)[:, None] * 2 * T + code[idx]).ravel()
    S = lambda v: np.bincount(flat, weights=np.asarray(v, float)[idx].ravel(), minlength=(B + 1) * 2 * T).reshape(B + 1, 2, T)
    h, n = S(sub["n_hits"]), S(sub["n_words"])
    wt = n.sum(1) / n.sum((1, 2))[:, None]
    r_t = np.where(n > 0, h / np.maximum(n, 1), np.nan) * 1000
    est = [np.nansum(r_t[:, i] * wt, 1) / np.nansum(np.where(np.isnan(r_t[:, i]), 0, wt), 1) for i in range(2)]
    dif = est[1] - est[0]
    print(f"\nTopic mix fixed, {typ} with harm clauses: before {boot_ci(est[0][1:], est[0][0])}  "
          f"after {boot_ci(est[1][1:], est[1][0])}  change {boot_ci(dif[1:], dif[0])}")

# === (4) again with the validated subset ==================================================
import statsmodels.api as sm
import statsmodels.formula.api as smf
reg = (items.drop(columns=["n_hits", "n_words"]).join(d[["n_hits", "n_words"]], how="inner")
            .join(it.set_index("row")[["label"]], how="inner"))
reg = reg[reg["n_words"] > 0].assign(log_words=lambda x: np.log(x["n_words"]))
reg["agent"] = pd.Categorical(reg["agent"], categories=["organisations", "ai_system", "agentless passive"])
reg["period"] = pd.Categorical(reg["chatgpt"], categories=["before", "after"])
reg["ctype"] = pd.Categorical(reg["content_type"], categories=["comment", "post"])
reg["topic"] = reg["label"].astype("category")
m = smf.glm("n_hits ~ agent + topic + period + ctype", data=reg,
            family=sm.families.Poisson(), offset=reg["log_words"]).fit(cov_type="HC1")
tab = pd.DataFrame({"IRR": np.exp(m.params), "lo": np.exp(m.conf_int()[0]), "hi": np.exp(m.conf_int()[1]), "p": m.pvalues})
print(f"=== Validated subset, n = {len(reg)} items ===")
print(tab.loc[[i for i in tab.index if i.startswith("agent") or i.startswith("period")]].round(3).to_string())
