# === 1. WHO IS TALKING? Author-level check (answers the previous reviewer) =============
# Does the shift come from new people arriving after ChatGPT, or do the same people
# talk differently? CI by resampling AUTHORS (clauses of one author are not independent).
d = load_items()
cl = pd.read_parquet(CLAUSES)
d["author"] = d["author"].astype(str).str.strip()
known = ~d["author"].isin(["", "None", "nan", "[deleted]", "[removed]"])
print(f"Items with a known author: {known.mean()*100:.1f}%")

per = d[known].groupby("author")["chatgpt"].agg(lambda s: set(s))
both = set(per[per.apply(lambda s: s == {"before", "after"})].index)
cl = cl.merge(d[["author"]].reset_index().rename(columns={"index": "row"}), on="row")
cl = cl[~cl["author"].isin(["", "None", "nan", "[deleted]", "[removed]"])]
cl["author_type"] = np.where(cl["author"].isin(both), "active in both periods",
                     np.where(cl["chatgpt"] == "before", "before only", "after only"))

print(f"\nAuthors active in both periods: {len(both):,} of {per.size:,}")
print("\n=== Harm clauses by author type and period ===")
print(pd.crosstab(cl["author_type"], cl["chatgpt"])[["before", "after"]].to_string())
after = cl[cl["chatgpt"] == "after"]
print(f"\nAfter ChatGPT, {(after['author_type'] == 'after only').mean()*100:.1f}% of harm clauses come "
      f"from authors who did not post before.")

rng = np.random.default_rng(3)
def by_author_boot(sub, groups, gcol, B=2000):
    """Shares of CATS in each group, CI by resampling authors."""
    authors = sub["author"].unique()
    pos = {a: i for i, a in enumerate(authors)}
    a_idx = sub["author"].map(pos).to_numpy()
    out = {}
    for g in groups:
        m = (sub[gcol] == g).to_numpy()
        for c in CATS:
            k = np.bincount(a_idx, weights=(m & (sub["grp"] == c).to_numpy()), minlength=len(authors))
            n = np.bincount(a_idx, weights=m.astype(float), minlength=len(authors))
            draw = rng.integers(0, len(authors), (B, len(authors)))
            est = k[draw].sum(1) / np.maximum(n[draw].sum(1), 1) * 100
            out[(c, g)] = boot_ci(est, k.sum() / n.sum() * 100)
    return pd.Series(out).unstack()

for name, sub in {"all verbs": cl, "without replace": cl[cl["verb"] != "replace"]}.items():
    print(f"\n######## {name}")
    stable = sub[sub["author_type"] == "active in both periods"]
    print(f"Same authors before and after (n = {len(stable):,} clauses, {stable['author'].nunique():,} authors):")
    print(by_author_boot(stable, ["before", "after"], "chatgpt").to_string())
    aft = sub[sub["chatgpt"] == "after"]
    print("\nAfter ChatGPT: returning vs new authors:")
    print(by_author_boot(aft, ["active in both periods", "after only"], "author_type").to_string())
    print("All authors (author-clustered CI):")
    print(by_author_boot(sub, ["before", "after"], "chatgpt").to_string())
cl[["row", "author_type"]].to_csv(f"{OUT}/author_types.csv", index=False)
print("\nIf the same authors show the same shift as everyone, population turnover does not explain it.")
