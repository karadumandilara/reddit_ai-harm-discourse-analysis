# === 2a. RECALL AND BASELINE SAMPLE: 200 sentences to code by hand =====================
# Stratum A (100): sentences that contain a word form of a listed harm verb.
#                  -> how many real harm clauses does the procedure miss (parser, object rules)?
# Stratum B (100): random sentences WITHOUT any listed verb form.
#                  -> how often is harm expressed with verbs outside the list?
# Half before, half after ChatGPT in each stratum.
VERBS = ("harm hurt injure kill damage ruin destroy endanger threaten discriminate misidentify "
         "misclassify mislabel profile target exclude deny flag censor shadowban fire replace displace "
         "arrest detain jail surveil monitor track spy collect harvest scrape leak expose sell steal "
         "scam defraud manipulate exploit deceive mislead misinform impersonate harass abuse violate "
         "invade silence").split()
def forms(v):
    """Regular inflections of a verb; irregular forms are added below."""
    if v.endswith("e"):
        return {v, v + "s", v + "d", v[:-1] + "ing"}
    if v.endswith("y"):
        return {v, v[:-1] + "ies", v[:-1] + "ied", v + "ing"}
    return {v, v + "s", v + "es", v + "ed", v + "ing"}
FORMS = set().union(*map(forms, VERBS)) | {"sold", "stole", "stolen", "misled", "scammed", "scamming",
         "flagged", "flagging", "shadowbanned", "surveilled", "surveilling", "targetted"}
VERB_RX = re.compile(r"\b(?:" + "|".join(sorted(FORMS, key=len, reverse=True)) + r")\b", re.I)

d = load_items()
cl = pd.read_parquet(CLAUSES)
def letters(s):
    return re.sub(r"[^a-z]", "", str(s).lower())
by_row = cl.groupby("row")["sentence"].apply(lambda x: [letters(s) for s in x]).to_dict()
def is_captured(row, s):
    """Same sentence as a clause sentence (spaCy and the regex split may cut sentences differently)."""
    k = letters(s)
    return any(c[:40] in k or k[:40] in c for c in by_row.get(row, []) if c and k)

rows_ = []
for i, t in d["text"].fillna("").items():
    for s in re.split(r"(?<=[.!?])\s+", t[:3000]):
        s = s.strip()
        if 5 <= len(s.split()) <= 80:
            rows_.append((i, s))
sent = pd.DataFrame(rows_, columns=["row", "sentence"])
sent["chatgpt"] = d.loc[sent["row"], "chatgpt"].to_numpy()
sent["has_verb"] = sent["sentence"].str.contains(VERB_RX)
print(f"{len(sent):,} sentences; {sent['has_verb'].mean()*100:.1f}% contain a listed verb form")

A = sent[sent["has_verb"]].groupby("chatgpt", group_keys=False).sample(50, random_state=41)
B = sent[~sent["has_verb"]].groupby("chatgpt", group_keys=False).sample(50, random_state=42)
samp = pd.concat([A.assign(stratum="A"), B.assign(stratum="B")]).sample(frac=1, random_state=43).reset_index(drop=True)
samp["id"] = np.arange(len(samp))
samp["captured"] = [int(is_captured(r, s)) for r, s in zip(samp["row"], samp["sentence"])]
samp[["id", "row", "stratum", "chatgpt", "captured"]].to_csv(f"{OUT}/recall_key.csv", index=False)
# The file to code does NOT show what the procedure found, so the coding is blind.
samp[["id", "sentence"]].assign(harm="", agent="").to_csv(f"{OUT}/recall_200_TO_CODE.csv", index=False)
pd.Series({"A": int(sent["has_verb"].sum()), "B": int((~sent["has_verb"]).sum())}).to_csv(f"{OUT}/recall_strata_sizes.csv")
print(f"Written {OUT}/recall_200_TO_CODE.csv")
print("""
Code each sentence:
  harm   1 = the sentence says that someone or something does harm to people or their data
             (same rules as validation: rights/privacy yes, rules/terms no, figurative no), else 0
  agent  only if harm = 1: who does the harm, one of
         ai_system, organisations, malicious, people, first_person, pronoun, no_agent, other
Save as recall_200.csv and run 2b.""")

# === RECALL, self-contained: rebuild the key from 2a (same seeds) and score your coding ====
import os, re
import numpy as np
import pandas as pd

CLEAN   = "/content/old_clean.parquet"
CLAUSES = "/content/harm_clauses_v3.parquet"
CODED   = "/content/recall_200.xlsx"            # <- your coded file (.xlsx or .csv; use Copy path)
for p in [CLEAN, CLAUSES, CODED]:
    print(("OK      " if os.path.exists(p) else "MISSING ") + p)

PERIODS = ["2020-2021", "2022-2023", "2024-2025"]
CHATGPT = pd.Timestamp("2022-11-30", tz="UTC")
d = pd.read_parquet(CLEAN)
d = d[d["period"].isin(PERIODS)].reset_index(drop=True)
d["chatgpt"] = np.where(pd.to_datetime(d["created_utc"], unit="s", utc=True) >= CHATGPT, "after", "before")
cl = pd.read_parquet(CLAUSES)

# --- 1. Rebuild the 2a sample exactly (same verb forms, same seeds) ---
VERBS = ("harm hurt injure kill damage ruin destroy endanger threaten discriminate misidentify "
         "misclassify mislabel profile target exclude deny flag censor shadowban fire replace displace "
         "arrest detain jail surveil monitor track spy collect harvest scrape leak expose sell steal "
         "scam defraud manipulate exploit deceive mislead misinform impersonate harass abuse violate "
         "invade silence").split()
def forms(v):
    if v.endswith("e"):
        return {v, v + "s", v + "d", v[:-1] + "ing"}
    if v.endswith("y"):
        return {v, v[:-1] + "ies", v[:-1] + "ied", v + "ing"}
    return {v, v + "s", v + "es", v + "ed", v + "ing"}
FORMS = set().union(*map(forms, VERBS)) | {"sold", "stole", "stolen", "misled", "scammed", "scamming",
         "flagged", "flagging", "shadowbanned", "surveilled", "surveilling", "targetted"}
VERB_RX = re.compile(r"\b(?:" + "|".join(sorted(FORMS, key=len, reverse=True)) + r")\b", re.I)
def letters(s):
    return re.sub(r"[^a-z]", "", str(s).lower())
by_row = cl.groupby("row")["sentence"].apply(lambda x: [letters(s) for s in x]).to_dict()
def is_captured(row, s):
    k = letters(s)
    return any(c[:40] in k or k[:40] in c for c in by_row.get(row, []) if c and k)
rows_ = []
for i, t in d["text"].fillna("").items():
    for s in re.split(r"(?<=[.!?])\s+", t[:3000]):
        s = s.strip()
        if 5 <= len(s.split()) <= 80:
            rows_.append((i, s))
sent = pd.DataFrame(rows_, columns=["row", "sentence"])
sent["chatgpt"] = d.loc[sent["row"], "chatgpt"].to_numpy()
sent["has_verb"] = sent["sentence"].str.contains(VERB_RX)
sizes = {"A": int(sent["has_verb"].sum()), "B": int((~sent["has_verb"]).sum())}
A = sent[sent["has_verb"]].groupby("chatgpt", group_keys=False).sample(50, random_state=41)
B = sent[~sent["has_verb"]].groupby("chatgpt", group_keys=False).sample(50, random_state=42)
key = pd.concat([A.assign(stratum="A"), B.assign(stratum="B")]).sample(frac=1, random_state=43).reset_index(drop=True)
key["id"] = np.arange(len(key))
key["captured"] = [int(is_captured(r, s)) for r, s in zip(key["row"], key["sentence"])]

# --- 2. Read your coding and check that the sentences match the rebuilt key ---
r = pd.read_excel(CODED) if CODED.endswith(".xlsx") else pd.read_csv(CODED, sep=None, engine="python")
r.columns = r.columns.str.strip().str.lower()
r = r.merge(key[["id", "row", "stratum", "chatgpt", "captured", "sentence"]].rename(columns={"sentence": "key_sentence"}), on="id")
match = (r["sentence"].map(letters).str[:40] == r["key_sentence"].map(letters).str[:40]).mean() * 100
print(f"\nSentences matching the rebuilt key: {match:.0f}%   (must be 100%; otherwise stop and tell me)")
r["harm"] = pd.to_numeric(r["harm"], errors="coerce")
r["agent"] = r["agent"].astype(str).str.strip().str.lower()
print("Empty harm codes:", r["harm"].isna().sum(), "of", len(r))

# --- 3. Recall ---
print("\n=== Harm sentences found by hand, by stratum ===")
print(r.groupby("stratum")["harm"].agg(coded="count", harm="sum", share="mean").round(3).to_string())
h = r[r["harm"] == 1]
rec_A = h.loc[h["stratum"] == "A", "captured"].mean()
print(f"\nRecall within sentences with a listed verb (stratum A): {rec_A*100:.1f}% "
      f"({h.loc[h['stratum']=='A','captured'].sum()} of {(h['stratum']=='A').sum()})")
pA, pB = r.loc[r["stratum"] == "A", "harm"].mean(), r.loc[r["stratum"] == "B", "harm"].mean()
hA_, hB_ = pA * sizes["A"], pB * sizes["B"]
print(f"Share of all harm sentences that contain a listed verb (stratum-weighted): {hA_/(hA_+hB_)*100:.1f}%")
print(f"Overall recall (stratum-weighted, rough): {rec_A * hA_/(hA_+hB_)*100:.1f}%")

# --- 4. Agent: procedure vs keyword baseline, on harm sentences the procedure captured ---
BASE = [("ai_system", r"\b(ai|algorithms?|models?|chatgpt|gpt\w*|llms?|bots?|software|apps?|cameras?|facial recognition|automation|robots?|machines?)\b"),
        ("organisations", r"\b(compan(y|ies)|corporations?|platforms?|google|meta|facebook|microsoft|openai|amazon|apple|tesla|government|police|cops?|state|agenc(y|ies)|employers?|bosses?|landlords?|banks?|insurers?|schools?)\b"),
        ("malicious", r"\b(hackers?|scammers?|criminals?|predators?|thie(f|ves)|bad actors?)\b"),
        ("people", r"\b(people|users?|humans?|workers?|customers?)\b")]
def baseline(s):
    s = s.lower(); best = (10**9, "none")
    for c, rx in BASE:
        m = re.search(rx, s)
        if m and m.start() < best[0]:
            best = (m.start(), c)
    return best[1]
grp_map = {"company": "organisations", "state": "organisations", "institution": "organisations",
           "developers": "organisations", "malicious_individuals": "malicious", "people_users": "people",
           "pronoun_unresolved": "pronoun", "no_agent (passive)": "no_agent"}
cl["proc"] = cl["agent_category"].replace(grp_map)
cl["k"] = cl["sentence"].map(letters)
def proc_agent(row, s):
    k = letters(s)
    for c, p in zip(cl.loc[cl["row"] == row, "k"], cl.loc[cl["row"] == row, "proc"]):
        if c[:40] in k or k[:40] in c:
            return p
    return "none"
hc = r[(r["harm"] == 1) & (r["captured"] == 1)].copy()
hc["procedure"] = [proc_agent(a, s) for a, s in zip(hc["row"], hc["sentence"])]
hc["baseline"] = hc["sentence"].apply(baseline)
print(f"\n=== Agent accuracy on captured harm sentences (n = {len(hc)}) ===")
print(f"procedure (dependency parse):         {(hc['procedure'] == hc['agent']).mean()*100:.1f}%")
print(f"keyword baseline (first actor term):  {(hc['baseline'] == hc['agent']).mean()*100:.1f}%")
print("\nMissed harm sentences (stratum A, not captured):")
print(r[(r["harm"] == 1) & (r["captured"] == 0) & (r["stratum"] == "A")][["id", "sentence"]].to_string(index=False))

# === Recall diagnostics: run right after the recall cell (uses r, hc) =====================
print("Agent codes you used:")
print(r.loc[r["harm"] == 1, "agent"].value_counts().to_string())
print("\nProcedure categories for the captured sentences:")
print(hc["procedure"].value_counts().to_string())
print("\n=== Captured harm sentences: your code vs procedure vs baseline ===")
pd.set_option("display.max_colwidth", 90)
print(hc[["id", "agent", "procedure", "baseline", "sentence"]].to_string(index=False))
print("\n=== Recall in stratum A by period ===")
hA = r[(r["harm"] == 1) & (r["stratum"] == "A")]
print(hA.groupby("chatgpt")["captured"].agg(harm_sentences="size", captured="sum", recall="mean").round(3).to_string())
print("\n=== Share of harm sentences by stratum and period ===")
print(r.groupby(["stratum", "chatgpt"])["harm"].agg(n="size", harm="sum", share="mean").round(3).to_string())

# === Agent accuracy with category codes for the captured harm sentences =====================
AGENT = {18: "no_agent", 21: "ai_system", 36: "no_agent", 48: "ai_system", 49: "organisations",
         64: "ai_system", 68: "organisations", 69: "people", 73: "other", 87: "ai_system",
         90: "other", 91: "pronoun", 93: "no_agent", 111: "other", 133: "ai_system",
         145: "ai_system", 170: "organisations", 178: "organisations", 185: "ai_system", 193: "pronoun"}
NOT_HARM = [19, 161]                      # recoded as not harm
x = hc[~hc["id"].isin(NOT_HARM)].copy()
x["agent"] = x["id"].map(AGENT)
print(f"n = {len(x)}")
print(f"procedure correct: {(x['procedure'] == x['agent']).sum()} ({(x['procedure'] == x['agent']).mean()*100:.0f}%)")
print(f"baseline correct:  {(x['baseline'] == x['agent']).sum()} ({(x['baseline'] == x['agent']).mean()*100:.0f}%)")
for c in ["ai_system", "organisations"]:
    t = x["agent"] == c
    print(f"  true {c:13s} n = {t.sum()}: procedure {((x['procedure'] == c) & t).sum()}, baseline {((x['baseline'] == c) & t).sum()}")

# === Recall again after recoding (uses r and sizes from the recall cell) ====================
NOT_HARM = [19, 161]          # add ids you recode to 0 after rechecking, e.g. 0, 1, 35, 106, 168, 199
rr = r.copy()
rr.loc[rr["id"].isin(NOT_HARM), "harm"] = 0
h = rr[(rr["harm"] == 1) & (rr["stratum"] == "A")]
rec = h["captured"].mean()
pA, pB = rr.loc[rr["stratum"] == "A", "harm"].mean(), rr.loc[rr["stratum"] == "B", "harm"].mean()
cov = pA * sizes["A"] / (pA * sizes["A"] + pB * sizes["B"])
print(f"Harm sentences: stratum A {int(rr.loc[rr['stratum']=='A','harm'].sum())} of 100, stratum B {int(rr.loc[rr['stratum']=='B','harm'].sum())} of 100")
print(f"Recall in stratum A: {rec*100:.1f}% ({int(h['captured'].sum())} of {len(h)})")
print(h.groupby("chatgpt")["captured"].agg(harm_sentences="size", captured="sum", recall="mean").round(3).to_string())
print(f"Share of all harm sentences containing a listed verb: {cov*100:.1f}%")
print(f"Overall recall (rough): {rec*cov*100:.1f}%")
print(f"Stratum sizes (sentences in the corpus): A {sizes['A']:,}, B {sizes['B']:,}")
