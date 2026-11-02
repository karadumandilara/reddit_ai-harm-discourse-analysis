# === 0. SETUP: run first, every session ===============================================
# Upload session_backup.zip (left panel > upload), then run this cell.
import os, re, zipfile
import numpy as np
import pandas as pd

BACKUP = "/content/session_backup.zip"
if os.path.exists(BACKUP) and not os.path.exists("/content/agency_v3/harm_clauses_v3.parquet"):
    zipfile.ZipFile(BACKUP).extractall("/content")
    print("Backup unpacked.")

CLEAN    = "/content/old_clean.parquet"
CLAUSES  = "/content/agency_v3/harm_clauses_v3.parquet"
CODED_V3 = "/content/agency_v3/validation_sample_v3_coded.csv"
TOPICS   = "/content/extra/item_topics.parquet"
LABELS   = "/content/extra/topic_labels.csv"
OUT      = "/content/extra2"
os.makedirs(OUT, exist_ok=True)
for p in [CLEAN, CLAUSES, CODED_V3, TOPICS, LABELS]:
    print(("OK      " if os.path.exists(p) else "MISSING ") + p)

PERIODS = ["2020-2021", "2022-2023", "2024-2025"]
CHATGPT = pd.Timestamp("2022-11-30", tz="UTC")
CATS = ("ai_system", "organisations")


def load_items():
    """Cleaned items in the same row order as the agency analysis (row = index)."""
    d = pd.read_parquet(CLEAN)
    d = d[d["period"].isin(PERIODS)].reset_index(drop=True)
    dt = pd.to_datetime(d["created_utc"], unit="s", utc=True)
    d["year"] = dt.dt.year
    d["chatgpt"] = np.where(dt >= CHATGPT, "after", "before")
    return d


def load_clauses_with_topics():
    cl = pd.read_parquet(CLAUSES)
    it = pd.read_parquet(TOPICS)
    lab = pd.read_csv(LABELS)
    ct = cl.merge(it, on="row").merge(lab[["topic", "label", "keep"]], on="topic", how="left")
    return ct[ct["keep"] == 1].copy()


def boot_ci(values, point, decimals=1):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return f"{point:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def read_any(path):
    """Read a CSV whatever its separator (comma, semicolon or tab)."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        sep = max(["\t", ";", ","], key=f.readline().count)
    x = pd.read_csv(path, sep=sep, encoding="utf-8-sig", encoding_errors="replace")
    x.columns = x.columns.str.strip()
    return x

print("Setup done. Output folder:", OUT)
