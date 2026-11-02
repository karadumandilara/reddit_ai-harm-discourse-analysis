# === 3. DECOMPOSITION: how much of the change is between topics, how much within? =====
# Kitagawa decomposition of the before/after change in a share across topics:
#   change = sum_t (w_after - w_before) * (p_before + p_after)/2     [composition]
#          + sum_t (p_after - p_before) * (w_before + w_after)/2     [within topics]
# w = topic weight (share of clauses), p = share of the agent category in the topic.
# 95% CI by resampling items. Only topics with clauses in both periods.
ct = load_clauses_with_topics()
rng = np.random.default_rng(17)

def decompose(sub, cat, B=2000):
    n_tp = sub.groupby(["label", "chatgpt"]).size().unstack(fill_value=0)
    topics = n_tp[(n_tp["before"] > 0) & (n_tp["after"] > 0)].index.tolist()
    sub = sub[sub["label"].isin(topics)]
    T = len(topics)
    it = sub.groupby("row").agg(per=("chatgpt", "first"), top=("label", "first"), n=("grp", "size"),
                                k=("grp", lambda g: (g == cat).sum()))
    code = (it["per"] == "after").to_numpy().astype(int) * T + it["top"].map({t: i for i, t in enumerate(topics)}).to_numpy()
    N = len(it)
    idx = np.vstack([np.arange(N), rng.integers(0, N, (B, N))])          # row 0 = point estimate
    flat = (np.arange(B + 1)[:, None] * 2 * T + code[idx]).ravel()
    def sums(v):
        return np.bincount(flat, weights=np.asarray(v, float)[idx].ravel(),
                           minlength=(B + 1) * 2 * T).reshape(B + 1, 2, T)
    n, k = sums(it["n"]), sums(it["k"])
    w = n / n.sum(axis=2, keepdims=True)
    p = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    comp = ((w[:, 1] - w[:, 0]) * (p[:, 0] + p[:, 1]) / 2) * 100        # (B+1, T)
    within = ((p[:, 1] - p[:, 0]) * (w[:, 0] + w[:, 1]) / 2) * 100
    C, W = comp.sum(1), within.sum(1)
    res = pd.Series({"total change": boot_ci(C[1:] + W[1:], C[0] + W[0]),
                     "composition (between topics)": boot_ci(C[1:], C[0]),
                     "within topics": boot_ci(W[1:], W[0]),
                     "share due to composition (%)": f"{C[0] / (C[0] + W[0]) * 100:.0f}" if (C[0] + W[0]) != 0 else "n/a"})
    contrib = pd.DataFrame({"composition": comp[0], "within": within[0]}, index=topics).round(2)
    return res, contrib

for name, sub in {"all verbs": ct, "without replace": ct[ct["verb"] != "replace"]}.items():
    for cat in CATS:
        res, contrib = decompose(sub, cat)
        print(f"\n=== {name} | {cat}: change in percentage points, 95% CI ===")
        print(res.to_string())
        print("Contribution by topic (pp):")
        print(contrib.reindex(contrib["composition"].abs().sort_values(ascending=False).index).to_string())
        contrib.to_csv(f"{OUT}/decomp_{cat}_{name.replace(' ', '_')}.csv")
