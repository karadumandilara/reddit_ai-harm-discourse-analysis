# === 10. EXPORT THE PUBLIC DATA FILES ===================================================
# Writes the files for the data/ folder of the repository. Reddit texts, usernames and
# sentence text are removed, as stated in the data availability statement: only
# identifiers, metadata, derived variables and codes are shared.
# Edit the paths below if your files have other names (left panel > right click > Copy path).
import os
import zipfile
import numpy as np
import pandas as pd

CLEAN    = "/content/old_clean.parquet"
CLAUSES  = "/content/harm_clauses_v3.parquet"
TOPICS   = "/content/item_topics.parquet"
LABELS   = "/content/topic_labels.csv"
VALID    = "/content/validation_sample_v3_coded.csv"   # second validation round (102 clauses)
RECALL   = "/content/recall_200.xlsx"                  # coded recall sample (200 sentences)
MFD150   = "/content/mfd_150.csv"                      # coded dictionary matches (after recoding)
EXTRA    = ["/content/extra2/topic_robustness.csv"]     # derived result tables to include as they are
OUT      = "/content/public_data"
os.makedirs(OUT, exist_ok=True)

# Columns that contain Reddit text or identify users. Anything matching is dropped.
TEXT_COLS = {"text", "body", "title", "selftext", "sentence", "context", "clean_text",
             "processed_text", "raw_text", "url", "permalink", "author", "author_fullname",
             "agent", "patient", "agent_text", "patient_text", "example_1", "example_2",
             "example_3", "representative_docs", "top_docs"}


def strip(df, name):
    drop = [c for c in df.columns if c.lower() in TEXT_COLS]
    print(f"{name}: dropped {drop}" if drop else f"{name}: nothing to drop")
    return df.drop(columns=drop)


def read_any(path):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


written = []

# 1. Items: identifiers and metadata of the 36,294 analysed items (row = position used in the analysis)
if os.path.exists(CLEAN):
    d = pd.read_parquet(CLEAN)
    d = d[d["period"].isin(["2020-2021", "2022-2023", "2024-2025"])].reset_index(drop=True)
    d.insert(0, "row", np.arange(len(d)))
    items = strip(d, "items")
    items.to_csv(f"{OUT}/items.csv", index=False); written.append("items.csv")

# 2. Harm clauses: verb, voice, agent category and period, without sentence text
if os.path.exists(CLAUSES):
    strip(pd.read_parquet(CLAUSES), "harm_clauses").to_csv(f"{OUT}/harm_clauses.csv", index=False)
    written.append("harm_clauses.csv")

# 3. Topic assignment and topic labels
if os.path.exists(TOPICS):
    pd.read_parquet(TOPICS).to_csv(f"{OUT}/item_topics.csv", index=False); written.append("item_topics.csv")
if os.path.exists(LABELS):
    strip(read_any(LABELS), "topic_labels").to_csv(f"{OUT}/topic_labels.csv", index=False)
    written.append("topic_labels.csv")

# 4. Validation data: codes only, linked to items by row or id
for path, name in [(VALID, "validation_102_codes.csv"), (RECALL, "recall_200_codes.csv"),
                   (MFD150, "mfd_150_codes.csv")]:
    if os.path.exists(path):
        strip(read_any(path), name).to_csv(f"{OUT}/{name}", index=False); written.append(name)
    else:
        print("MISSING", path)

# 5. Derived result tables
for path in EXTRA:
    if os.path.exists(path):
        pd.read_csv(path).to_csv(f"{OUT}/{os.path.basename(path)}", index=False)
        written.append(os.path.basename(path))

print("\nWritten to", OUT, ":", written)
print("\nColumns kept per file (check that no text or usernames remain):")
for f in written:
    print(f"  {f}: {list(pd.read_csv(f'{OUT}/{f}', nrows=1).columns)}")

zpath = "/content/public_data.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in written:
        z.write(f"{OUT}/{f}", f)
print("\nZip:", zpath)
try:
    from google.colab import files
    files.download(zpath)
except ImportError:
    pass
