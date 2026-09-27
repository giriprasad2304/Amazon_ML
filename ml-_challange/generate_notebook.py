#!/usr/bin/env python3
"""Generates Amazon_ML_Challenge_Colab.ipynb from the inlined src/ code."""
import json, textwrap

def cell(source, kind="code"):
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(cell("""\
# 🏆 Amazon ML Challenge 2026 — Business Entity Resolution

**Task**: For every Source 1 business, find all Source 2 / Source 3 records that describe the same real-world business.  
**Metric**: Macro-averaged F0.5 (precision-weighted) per Source 1 entity.

### Pipeline overview
```
Dataset Upload → Cache Build → Text Normalization
  → Blocking (5 key types, IDF-ranked) → 34 Pair Features
    → HistGradientBoosting (E4) → Decision + Conflict Resolution
      → matching_results.tsv  +  candidate_pairs.tsv
```

### Final model: E4
- Transliteration of Indian scripts (`anyascii`)
- K=100 candidates per entity
- 32 fast features (no slow TF-IDF)
- 40 000 training entities
- **Validation F0.5 ≈ 0.9421**

> **This is a single-file notebook** — every helper from the original `src/` package is inlined.  
> Upload your `dataset/` folder (TSVs from Kaggle) and run all cells top-to-bottom.\
""", "markdown"))

# ── SECTION 1: INSTALL ────────────────────────────────────────────────────────
cells.append(cell("---\n## 📦 Section 1 — Install dependencies", "markdown"))
cells.append(cell("""\
import subprocess, sys
packages = [
    "pandas==3.0.3", "numpy==2.4.6", "scikit-learn==1.9.0",
    "pyarrow==23.0.1", "rapidfuzz==3.14.6", "anyascii==0.3.3",
    "joblib", "psutil",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
print("✅ All packages installed.")\
"""))

# ── SECTION 2: DATASET UPLOAD ─────────────────────────────────────────────────
cells.append(cell("---\n## 📁 Section 2 — Upload / Mount dataset\n\n**Option A** — Google Drive (recommended for large files)  \n**Option B** — Direct ZIP upload  \n\nAfter upload, set `DATASET_ROOT` to the folder containing `train/` and `test/`.", "markdown"))
cells.append(cell("""\
USE_DRIVE = True   # Set False to use direct upload

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    # ↓ Change to the path inside your Drive
    DATASET_ROOT = "/content/drive/MyDrive/amazon_ml_challenge/dataset"
else:
    from google.colab import files
    import zipfile
    print("Upload your dataset ZIP (should contain train/ and test/):")
    uploaded = files.upload()
    zip_path = list(uploaded.keys())[0]
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("/content/dataset")
    DATASET_ROOT = "/content/dataset"

import pathlib
DATASET_ROOT = pathlib.Path(DATASET_ROOT)
TRAIN_DIR    = DATASET_ROOT / "train"
TEST_DIR     = DATASET_ROOT / "test"
assert TRAIN_DIR.exists(), f"train/ not found under {DATASET_ROOT}"
assert TEST_DIR.exists(),  f"test/ not found under {DATASET_ROOT}"
print(f"✅ Dataset root: {DATASET_ROOT}")
print("   train:", sorted(p.name for p in TRAIN_DIR.iterdir()))
print("   test: ", sorted(p.name for p in TEST_DIR.iterdir()))\
"""))
cells.append(cell("""\
import pathlib, os

PROJECT_ROOT = pathlib.Path("/content/amazon_ml")
CACHE_DIR    = PROJECT_ROOT / "cache"
OUTPUT_DIR   = PROJECT_ROOT / "output"
MODELS_DIR   = CACHE_DIR / "models"

for d in [CACHE_DIR, OUTPUT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Directories:", PROJECT_ROOT, CACHE_DIR, OUTPUT_DIR, MODELS_DIR)\
"""))

# ── SECTION 3: HELPER MODULES ─────────────────────────────────────────────────
cells.append(cell("---\n## 🔧 Section 3 — Inlined helper modules\n\nAll code from `src/` defined here as plain functions — no `import src.*` needed.", "markdown"))

# data_loading
cells.append(cell("""\
# ═══════════════════════════════════════════════════
# DATA LOADING  (src/data_loading.py)
# ═══════════════════════════════════════════════════
import csv, pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\\t", dtype="string[pyarrow]",
                       keep_default_na=False, quoting=csv.QUOTE_NONE, usecols=usecols)

def load_source(split, source_number, usecols=None):
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    return read_tsv(folder / f"{split}_source{source_number}.tsv", usecols=usecols)

def load_ground_truth():
    return read_tsv(TRAIN_DIR / "train_ground_truth.tsv")

def ground_truth_to_pairs(ground_truth):
    has_match = ground_truth[ground_truth["matched_entity_ids"] != ""]
    pairs = has_match.assign(match_id=has_match["matched_entity_ids"].str.split(",")).explode("match_id")
    pairs = pairs[["source1_entity_id", "match_id"]].reset_index(drop=True)
    pairs["match_id"] = pairs["match_id"].astype("string[pyarrow]")
    pairs["match_source"] = pairs["match_id"].str[:2]
    return pairs

print("✅ Data-loading helpers defined.")\
"""))

# preprocessing
cells.append(cell(r'''# ═══════════════════════════════════════════════════
# TEXT NORMALIZATION  (src/preprocessing.py)
# ═══════════════════════════════════════════════════
import pandas as pd

INDIAN_SCRIPTS = "[ऀ-ൿ]"

def transliterate(text):
    from anyascii import anyascii
    has = text.str.contains(INDIAN_SCRIPTS)
    if not has.any(): return text
    out = text.copy()
    out[has] = [anyascii(v).lower() for v in text[has]]
    return out

TRADE_NAME_MARKER = r"^.+?\s(?:d\.?b\.?a\.?|a\.?k\.?a\.?|née|formerly known as|formerly|t/a|trading as):?\s+"

def keep_real_name(text):    return text.str.replace(TRADE_NAME_MARKER, "", regex=True)
def remove_invisible(text):  return text.str.replace("[​-‍﻿­]", "", regex=True)
def strip_accents(text):     return text.str.normalize("NFKD").str.replace("[̀-ͯ]", "", regex=True)
def ampersand_to_and(text):  return text.str.replace("&", " and ", regex=False)
def drop_web(text):
    text = text.str.replace(r"\bwww\.", "", regex=True)
    return text.str.replace(r"\.(?:com|net|org|co\.in|in|co|fr)\b", " ", regex=True)
def join_dotted(text):       return text.str.replace(".", "", regex=False)

def replace_words(text, mapping):
    text = " " + text + " "
    for word, replacement in mapping.items():
        new = f" {replacement} " if replacement else " "
        text = text.str.replace(f" {word} ", new, regex=False).str.replace(f" {word} ", new, regex=False)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()

LEGAL_CANONICAL = {
    "pvt": "private", "pvt ltd": "private limited", "ltd": "limited",
    "inc": "incorporated", "corp": "corporation", "co": "company",
    "l l c": "llc", "l l p": "llp", "p c": "pc",
}
LEGAL_WORDS = [
    "private", "pvt", "limited", "ltd", "public", "llc", "llp", "pllc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "pc", "lp", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei",
]
ADDRESS_ABBREVIATIONS = {
    "st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "ln": "lane",
    "blvd": "boulevard", "bd": "boulevard", "ct": "court", "cir": "circle", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "apt": "apartment", "r": "rue",
}

def canonical_legal(text):             return replace_words(text, LEGAL_CANONICAL)
def remove_legal(text):                return replace_words(text, {w: "" for w in LEGAL_WORDS})
def expand_address_abbreviations(text):return replace_words(text, ADDRESS_ABBREVIATIONS)
def drop_placeholders(text):           return replace_words(text, {"null": ""})
def strip_leading_zeros(text):         return text.str.replace(r"\b0+(\d)", r"\1", regex=True)

RAW_STEPS  = {"translit": transliterate, "trade_name": keep_real_name,
               "invisible": remove_invisible, "accents": strip_accents,
               "ampersand": ampersand_to_and, "web": drop_web, "dots": join_dotted}
WORD_STEPS = {"legal_canonical": canonical_legal, "legal_remove": remove_legal,
               "address_abbrev": expand_address_abbreviations,
               "null": drop_placeholders, "zeros": strip_leading_zeros}
STEP_ORDER = list(RAW_STEPS) + list(WORD_STEPS)

NAME_STEPS    = ("trade_name", "invisible", "accents", "web")
ADDRESS_STEPS = ("invisible", "accents", "address_abbrev", "null", "zeros")

def basic_clean(text):
    return (text.str.replace(r"[^\p{L}\p{N}\p{M}\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True).str.strip())

def normalize(text, steps=()):
    unknown = set(steps) - set(STEP_ORDER)
    if unknown: raise ValueError(f"unknown steps: {unknown}")
    text = text.str.lower()
    for step in RAW_STEPS:
        if step in steps: text = RAW_STEPS[step](text)
    text = basic_clean(text)
    for step in WORD_STEPS:
        if step in steps: text = WORD_STEPS[step](text)
    return text

print("✅ Preprocessing helpers defined.")'''))

# splits + evaluation
cells.append(cell("""\
# ═══════════════════════════════════════════════════
# SPLITS  (src/splits.py)
# ═══════════════════════════════════════════════════
import pandas as pd

VALIDATION_FRACTION   = 0.20
SMALL_VALIDATION_SIZE = 20_000
SEED = 42

def make_validation_split(ground_truth):
    all_ids              = ground_truth["source1_entity_id"].sort_values().reset_index(drop=True)
    validation_ids       = all_ids.sample(frac=VALIDATION_FRACTION, random_state=SEED)
    small_validation_ids = validation_ids.sample(n=SMALL_VALIDATION_SIZE, random_state=SEED)
    split = pd.DataFrame({"source1_entity_id": all_ids})
    split["split"] = "train"
    split.loc[split["source1_entity_id"].isin(validation_ids), "split"] = "validation"
    split["small_validation"] = split["source1_entity_id"].isin(small_validation_ids)
    return split

print("✅ Split helpers defined.")

# ═══════════════════════════════════════════════════
# EVALUATION  (src/evaluation.py)
# ═══════════════════════════════════════════════════

def to_id_sets(id_strings):
    return id_strings.fillna("").map(lambda t: set(t.split(",")) if t else set())

def entity_f05(true_ids, predicted_ids):
    if not true_ids and not predicted_ids: return 1.0
    correct = len(true_ids & predicted_ids)
    if correct == 0: return 0.0
    precision = correct / len(predicted_ids)
    recall    = correct / len(true_ids)
    return 1.25 * precision * recall / (0.25 * precision + recall)

def score_per_entity(predictions, truth):
    table = truth[["source1_entity_id", "matched_entity_ids"]].merge(
        predictions[["source1_entity_id", "matched_entity_ids"]],
        on="source1_entity_id", how="left", suffixes=("_true", "_pred"))
    true_sets = to_id_sets(table["matched_entity_ids_true"])
    pred_sets = to_id_sets(table["matched_entity_ids_pred"])
    table["n_true"]    = true_sets.map(len)
    table["n_pred"]    = pred_sets.map(len)
    table["n_correct"] = [len(t & p) for t, p in zip(true_sets, pred_sets)]
    table["f05"]       = [entity_f05(t, p) for t, p in zip(true_sets, pred_sets)]
    return table

def macro_f05(predictions, truth):
    return score_per_entity(predictions, truth)["f05"].mean()

def score_report(predictions, truth):
    table      = score_per_entity(predictions, truth)
    singletons = table["n_true"] == 0
    total_pred = table["n_pred"].sum()
    total_true = table["n_true"].sum()
    return pd.Series({
        "macro_F0.5":                    table["f05"].mean(),
        "F0.5 on entities with matches": table.loc[~singletons, "f05"].mean(),
        "F0.5 on singletons":            table.loc[ singletons, "f05"].mean(),
        "pair precision":  table["n_correct"].sum() / total_pred if total_pred else float("nan"),
        "pair recall":     table["n_correct"].sum() / total_true if total_true else float("nan"),
        "predicted pairs": total_pred,
        "true pairs":      total_true,
    })

def candidate_recall(candidates, truth):
    renamed = candidates.rename(columns={"candidate_entity_ids": "matched_entity_ids"})
    table   = score_per_entity(renamed, truth)
    return table["n_correct"].sum() / table["n_true"].sum()

print("✅ Evaluation helpers defined.")\
"""))

# blocking keys
cells.append(cell("""\
# ═══════════════════════════════════════════════════
# BLOCKING KEYS  (src/blocking_keys.py)
# ═══════════════════════════════════════════════════
import numpy as np, pandas as pd

ALL_KEY_TYPES  = ["nw", "aw", "nb", "ab", "nj"]
NAME_KEY_TYPES = {"nw", "nb", "nj"}
_REMOVE_LEGAL  = {w: "" for w in LEGAL_WORDS}

def _word_rows(text):
    words = text.reset_index(drop=True).str.split().explode()
    return words[words.notna() & (words != "")]

def _bigram_rows(text):
    words     = _word_rows(text)
    following = words.groupby(level=0).shift(-1)
    keep      = following.notna()
    return words[keep] + "_" + following[keep]

def _joined_rows(name):
    joined = replace_words(name.reset_index(drop=True), _REMOVE_LEGAL).str.replace(" ", "", regex=False)
    return joined[joined.str.len() >= 4]

def record_keys(names, addresses, countries, key_types=ALL_KEY_TYPES):
    makers = {
        "nw": lambda: _word_rows(names),
        "aw": lambda: _word_rows(addresses),
        "nb": lambda: _bigram_rows(names),
        "ab": lambda: _bigram_rows(addresses),
        "nj": lambda: _joined_rows(names),
    }
    countries = countries.reset_index(drop=True)
    parts = []
    for key_type in key_types:
        values = makers[key_type]()
        if values.empty: continue
        keys = countries.reindex(values.index) + f"|{key_type}|" + values.astype("str")
        parts.append(pd.DataFrame({
            "row":     values.index.to_numpy(),
            "key":     keys.to_numpy(),
            "is_name": np.int8(key_type in NAME_KEY_TYPES),
        }))
    return pd.concat(parts, ignore_index=True)

print("✅ Blocking-key helpers defined.")\
"""))

# blocking
cells.append(cell("""\
# ═══════════════════════════════════════════════════
# BLOCKING  (src/blocking.py)
# ═══════════════════════════════════════════════════
import gc, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq

def build_index(s1, pool_path, max_df, key_types=ALL_KEY_TYPES, chunk_rows=1_000_000, log=print):
    t0 = time.time()
    q          = record_keys(s1["name_norm"], s1["addr_norm"], s1["country"], key_types)
    vocabulary = pd.Index(q["key"].unique())
    is_name_key = np.asarray(vocabulary.str.split("|").str[1].isin(NAME_KEY_TYPES))
    q = pd.DataFrame({"s1_row": q["row"].astype("int32"),
                       "code":   vocabulary.get_indexer(q["key"]).astype("int32")}).drop_duplicates()
    df = np.zeros(len(vocabulary), dtype=np.int64)
    pool_ids, pool_keys, offset = [], [], 0
    for batch in pq.ParquetFile(pool_path).iter_batches(batch_size=chunk_rows,
            columns=["entity_id", "country", "name_norm", "addr_norm"]):
        chunk = batch.to_pandas()
        pool_ids.append(chunk["entity_id"])
        k     = record_keys(chunk["name_norm"], chunk["addr_norm"], chunk["country"], key_types)
        codes = vocabulary.get_indexer(k["key"]); keep = codes >= 0
        k = pd.DataFrame({"pool_row": (k["row"].to_numpy()[keep] + offset).astype("int32"),
                           "code":     codes[keep].astype("int32")}).drop_duplicates()
        cc = np.bincount(k["code"], minlength=len(vocabulary))
        df += cc
        pool_keys.append(k[cc[k["code"]] <= max_df])
        offset += len(chunk)
    pool_ids  = pd.concat(pool_ids,  ignore_index=True)
    pool_keys = pd.concat(pool_keys, ignore_index=True)
    pool_keys = pool_keys[df[pool_keys["code"]] <= max_df]
    q         = q[df[q["code"]] <= max_df]
    gc.collect()
    log(f"  index built: {len(q):,} query keys, {len(pool_keys):,} pool keys ({time.time()-t0:.0f}s)")
    return {"query_keys": q, "pool_keys": pool_keys,
            "idf": np.log(len(pool_ids) / np.maximum(df, 1)),
            "is_name_key": is_name_key, "pool_ids": pool_ids}

def generate_candidates(s1, index, top_k=50, ranking="all", batch_size=1_000):
    q, pool_keys   = index["query_keys"], index["pool_keys"]
    idf, is_name_key = index["idf"], index["is_name_key"]
    kept = []
    for b in range(0, len(s1), batch_size):
        part = q[(q["s1_row"] >= b) & (q["s1_row"] < b + batch_size)].merge(pool_keys, on="code")
        part["w"]      = idf[part["code"]]
        part["w_name"] = np.where(is_name_key[part["code"]], part["w"], 0.0)
        pair = part.groupby(["s1_row", "pool_row"]).agg(
            score_all=("w", "sum"), score_name=("w_name", "sum")).reset_index()
        pair["score_addr"] = pair["score_all"] - pair["score_name"]
        if ranking == "all":
            keep = pair.groupby("s1_row")["score_all"].rank(method="first", ascending=False) <= top_k
        else:
            keep = ((pair.groupby("s1_row")["score_name"].rank(method="first", ascending=False) <= top_k//2) |
                    (pair.groupby("s1_row")["score_addr"].rank(method="first", ascending=False) <= top_k//2))
        kept.append(pair[keep])
    candidates = pd.concat(kept, ignore_index=True)
    candidates["source1_entity_id"] = s1["entity_id"].to_numpy()[candidates["s1_row"]]
    candidates["candidate_id"]      = index["pool_ids"].to_numpy()[candidates["pool_row"]]
    return candidates

def _hash(keys):
    return pd.util.hash_pandas_object(pd.Series(keys), index=False).to_numpy()

def build_index_hashed(s1, pool_batches, n_pool, max_df, key_types=ALL_KEY_TYPES,
                        query_chunk=100_000, log=print):
    t0 = time.time()
    q_parts = []
    for first in range(0, len(s1), query_chunk):
        part = s1.iloc[first:first + query_chunk]
        k    = record_keys(part["name_norm"], part["addr_norm"], part["country"], key_types)
        q_parts.append(pd.DataFrame({"s1_row": (k["row"].to_numpy() + first).astype("int32"),
                                      "h": _hash(k["key"]), "is_name": k["is_name"].to_numpy()}))
    q = pd.concat(q_parts, ignore_index=True).drop_duplicates(["s1_row", "h"])
    vocabulary, inverse = np.unique(q["h"].to_numpy(), return_inverse=True)
    q["code"] = inverse.astype("int32")
    is_name_key = np.zeros(len(vocabulary), dtype=bool)
    is_name_key[q["code"].to_numpy()] = q["is_name"].to_numpy().astype(bool)
    q = q[["s1_row", "code"]]
    df = np.zeros(len(vocabulary), dtype=np.int64)
    pool_ids, pool_keys, offset = [], [], 0
    for chunk in pool_batches:
        chunk = chunk.reset_index(drop=True)
        pool_ids.append(chunk["entity_id"])
        k = record_keys(chunk["name_norm"], chunk["addr_norm"], chunk["country"], key_types)
        h = _hash(k["key"])
        position = np.searchsorted(vocabulary, h)
        position[position == len(vocabulary)] = 0
        hit = vocabulary[position] == h
        k = pd.DataFrame({"pool_row": (k["row"].to_numpy()[hit] + offset).astype("int32"),
                           "code":     position[hit].astype("int32")}).drop_duplicates()
        cc = np.bincount(k["code"], minlength=len(vocabulary))
        df += cc; pool_keys.append(k[cc[k["code"]] <= max_df])
        offset += len(chunk)
    pool_ids  = pd.concat(pool_ids,  ignore_index=True)
    pool_keys = pd.concat(pool_keys, ignore_index=True)
    pool_keys = pool_keys[df[pool_keys["code"]] <= max_df]
    q         = q[df[q["code"]] <= max_df]
    gc.collect()
    log(f"  hashed index: {len(s1):,} queries, {len(q):,} query keys ({time.time()-t0:.0f}s)")
    return {"query_keys": q, "pool_keys": pool_keys,
            "idf": np.log(n_pool / np.maximum(df, 1)),
            "is_name_key": is_name_key, "pool_ids": pool_ids}

def iter_candidates(s1, index, top_k=50, batch_size=5_000, yield_every=10_000):
    idf, is_name_key = index["idf"], index["is_name_key"]
    pool_keys   = index.pop("pool_keys")
    pool_codes  = pool_keys["code"].to_numpy()
    order       = np.argsort(pool_codes, kind="stable")
    pool_rows_sorted = pool_keys["pool_row"].to_numpy()[order]
    offsets     = np.searchsorted(pool_codes[order], np.arange(len(idf) + 1))
    del pool_keys, pool_codes, order; gc.collect()
    q         = index["query_keys"].sort_values("s1_row")
    q_rows, q_codes = q["s1_row"].to_numpy(), q["code"].to_numpy()
    starts    = np.searchsorted(q_rows, np.arange(0, len(s1) + batch_size, batch_size))
    kept = []
    for i, first in enumerate(range(0, len(s1), batch_size)):
        rows, codes = q_rows[starts[i]:starts[i+1]], q_codes[starts[i]:starts[i+1]]
        counts = offsets[codes + 1] - offsets[codes]; total = int(counts.sum())
        if total:
            begin    = np.repeat(offsets[codes], counts)
            within   = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            code_rep = np.repeat(codes, counts); weight = idf[code_rep]
            part = pd.DataFrame({"s1_row": np.repeat(rows, counts),
                                  "pool_row": pool_rows_sorted[begin + within],
                                  "w": weight, "w_name": np.where(is_name_key[code_rep], weight, 0.0)})
            pair = part.groupby(["s1_row", "pool_row"]).agg(
                score_all=("w", "sum"), score_name=("w_name", "sum")).reset_index()
            pair["score_addr"] = pair["score_all"] - pair["score_name"]
            kept.append(pair[pair.groupby("s1_row")["score_all"].rank(method="first", ascending=False) <= top_k])
        if (first + batch_size) % yield_every == 0 or first + batch_size >= len(s1):
            if kept:
                candidates = pd.concat(kept, ignore_index=True)
                candidates["source1_entity_id"] = s1["entity_id"].to_numpy()[candidates["s1_row"]]
                candidates["candidate_id"]      = index["pool_ids"].to_numpy()[candidates["pool_row"]]
                yield candidates
            kept = []

print("✅ Blocking helpers defined.")\
"""))

# features
cells.append(cell(r'''# ═══════════════════════════════════════════════════
# PAIR FEATURES  (src/features.py)
# ═══════════════════════════════════════════════════
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

LEGAL_FORM = {"pvt": "private", "ltd": "limited", "inc": "incorporated", "corp": "corporation", "co": "company"}
LEGAL_SET  = set(LEGAL_WORDS)

def pairwise(scorer, left, right):
    return process.cpdist(left.tolist(), right.tolist(), scorer=scorer, workers=-1) / 100.0

def jaccard_lists(left, right):
    out = np.full(len(left), np.nan, dtype="float32")
    for i, (a, b) in enumerate(zip(left, right)):
        a, b = set(a), set(b)
        if a and b: out[i] = len(a & b) / len(a | b)
    return out

def legal_forms(name_words):
    return [frozenset(LEGAL_FORM.get(w, w) for w in words if w in LEGAL_SET) for words in name_words]

def fit_tfidf(texts, max_features=200_000):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), min_df=2,
                        max_features=max_features, dtype=np.float32)
    v.fit(texts); return v

def tfidf_cosine(vectorizer, left, right):
    ul, il = np.unique(left.to_numpy(dtype=object),  return_inverse=True)
    ur, ir = np.unique(right.to_numpy(dtype=object), return_inverse=True)
    a = vectorizer.transform(ul)[il]; b = vectorizer.transform(ur)[ir]
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype("float32")

def _core(word_lists):
    return pd.Series([" ".join(w for w in words if w not in LEGAL_SET) for words in word_lists], dtype="str")

def add_features(pairs, name_vectorizer=None, addr_vectorizer=None):
    f = pairs.reset_index(drop=True)
    nsw = f["name_s1"].str.split(); ncw = f["name_c"].str.split()
    cs1 = _core(nsw); cc = _core(ncw)
    f["name_exact"]       = (f["name_s1"] == f["name_c"]).astype("int8")
    f["name_jaccard"]     = jaccard_lists(nsw, ncw)
    f["name_ratio"]       = pairwise(fuzz.ratio,          f["name_s1"], f["name_c"])
    f["name_token_sort"]  = pairwise(fuzz.token_sort_ratio, f["name_s1"], f["name_c"])
    f["name_token_set"]   = pairwise(fuzz.token_set_ratio,  f["name_s1"], f["name_c"])
    f["core_ratio"]       = pairwise(fuzz.ratio,          cs1, cc)
    f["core_token_set"]   = pairwise(fuzz.token_set_ratio, cs1, cc)
    f["core_partial"]     = pairwise(fuzz.partial_ratio,   cs1, cc)
    f["core_joined_ratio"]= pairwise(fuzz.ratio, cs1.str.replace(" ","",regex=False), cc.str.replace(" ","",regex=False))
    if name_vectorizer is not None:
        f["name_tfidf"] = tfidf_cosine(name_vectorizer, f["name_s1"], f["name_c"])
    f["name_len_diff"]         = (f["name_s1"].str.len() - f["name_c"].str.len()).abs().astype("float32")
    f["name_c_foreign_script"] = (f["name_c_foreign"].astype("int8") if "name_c_foreign" in f
                                   else f["name_c"].str.contains("[ऀ-ൿ]").astype("int8"))
    fs1, fc = legal_forms(nsw), legal_forms(ncw)
    f["legal_same"]        = np.array([bool(a) and a==b          for a,b in zip(fs1,fc)], dtype="int8")
    f["legal_conflict"]    = np.array([bool(a) and bool(b) and not (a&b) for a,b in zip(fs1,fc)], dtype="int8")
    f["legal_one_missing"] = np.array([bool(a) != bool(b)         for a,b in zip(fs1,fc)], dtype="int8")
    f["addr_c_empty"]   = (f["addr_c"] == "").astype("int8")
    f["addr_jaccard"]   = jaccard_lists(f["addr_s1"].str.split(), f["addr_c"].str.split())
    f["addr_ratio"]     = pairwise(fuzz.ratio,           f["addr_s1"], f["addr_c"])
    f["addr_token_set"] = pairwise(fuzz.token_set_ratio, f["addr_s1"], f["addr_c"])
    f["addr_token_sort"]= pairwise(fuzz.token_sort_ratio,f["addr_s1"], f["addr_c"])
    if addr_vectorizer is not None:
        f["addr_tfidf"] = tfidf_cosine(addr_vectorizer, f["addr_s1"], f["addr_c"])
    for col in [c for c in ["addr_ratio","addr_token_set","addr_token_sort","addr_tfidf"] if c in f]:
        f.loc[f["addr_c_empty"]==1, col] = np.nan
    ns1=[set(x) for x in f["addr_s1"].str.findall(r"\d+")]
    nc =[set(x) for x in f["addr_c"].str.findall(r"\d+")]
    f["num_shared"]   = np.array([len(a&b) for a,b in zip(ns1,nc)], dtype="float32")
    f["num_jaccard"]  = np.array([len(a&b)/len(a|b) if a and b else np.nan for a,b in zip(ns1,nc)], dtype="float32")
    f["num_conflict"] = np.array([bool(a) and bool(b) and not (a&b) for a,b in zip(ns1,nc)], dtype="int8")
    f["num_c_subset"] = np.array([bool(b) and b<=a for a,b in zip(ns1,nc)], dtype="int8")
    g = f.groupby("s1_row")
    f["block_score"]        = f["score_all"].astype("float32")
    f["block_score_name"]   = f["score_name"].astype("float32")
    f["block_score_addr"]   = f["score_addr"].astype("float32")
    f["block_rank"]         = g["score_all"].rank(method="first", ascending=False).astype("float32")
    f["block_score_rel"]    = (f["score_all"] / g["score_all"].transform("max")).astype("float32")
    f["n_candidates"]       = g["score_all"].transform("size").astype("float32")
    f["name_token_set_rel"] = (f["name_token_set"] - g["name_token_set"].transform("max")).astype("float32")
    f["addr_token_set_rel"] = (f["addr_token_set"].fillna(0) - g["addr_token_set"].transform("max").fillna(0)).astype("float32")
    f["is_source3"]         = f["candidate_id"].str.startswith("S3").astype("int8")
    return f

FEATURE_COLUMNS = [
    "name_exact","name_jaccard","name_ratio","name_token_sort","name_token_set",
    "core_ratio","core_token_set","core_partial","core_joined_ratio",
    "name_tfidf","name_len_diff","name_c_foreign_script",
    "legal_same","legal_conflict","legal_one_missing",
    "addr_c_empty","addr_jaccard","addr_ratio","addr_token_set","addr_token_sort","addr_tfidf",
    "num_shared","num_jaccard","num_conflict","num_c_subset",
    "block_score","block_score_name","block_score_addr","block_rank","block_score_rel","n_candidates",
    "name_token_set_rel","addr_token_set_rel","is_source3",
]
FAST_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("name_tfidf","addr_tfidf")]

print("✅ Feature helpers defined.")'''))

# decision + pipeline
cells.append(cell("""\
# ═══════════════════════════════════════════════════
# DECISION RULES  (src/decision.py)
# ═══════════════════════════════════════════════════
import numpy as np, pandas as pd

def global_threshold(pairs, threshold):
    return pairs[pairs["probability"] >= threshold]

def resolve_conflicts(matches):
    best = matches.groupby("candidate_id")["probability"].transform("max")
    return matches[matches["probability"] == best].drop_duplicates("candidate_id")

def to_submission(matches, column="matched_entity_ids"):
    return matches.groupby("source1_entity_id")["candidate_id"].agg(",".join).rename(column).reset_index()

print("✅ Decision helpers defined.")

# ═══════════════════════════════════════════════════
# PIPELINE UTILITIES  (src/pipeline.py)
# ═══════════════════════════════════════════════════
import gc, hashlib, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

def normalization_tag(name_steps, address_steps):
    text = "+".join(name_steps) + "|" + "+".join(address_steps)
    return hashlib.md5(text.encode()).hexdigest()[:8]

def normalized_pool(split, name_steps, address_steps, log=print):
    path = CACHE_DIR / f"{split}_pool_norm_{normalization_tag(name_steps, address_steps)}.parquet"
    if path.exists(): return path
    t0 = time.time(); writer = None
    for batch in pq.ParquetFile(CACHE_DIR / f"{split}_pool.parquet").iter_batches(batch_size=1_000_000):
        chunk = batch.to_pandas()
        out   = pd.DataFrame({
            "entity_id":    chunk["entity_id"], "country": chunk["country"],
            "name_norm":    normalize(chunk["business_name"], name_steps),
            "addr_norm":    normalize(chunk["business_address"], address_steps),
            "name_foreign": chunk["business_name"].str.contains(INDIAN_SCRIPTS).astype("int8"),
        })
        table  = pa.Table.from_pandas(out, preserve_index=False)
        writer = writer or pq.ParquetWriter(path, table.schema)
        writer.write_table(table, row_group_size=1_000_000)
    writer.close()
    log(f"  normalized {split} pool cached ({time.time()-t0:.0f}s): {path.name}")
    return path

def prepare_queries(s1, name_steps, address_steps):
    s1 = s1.reset_index(drop=True).copy()
    s1["name_norm"] = normalize(s1["business_name"], name_steps)
    s1["addr_norm"] = normalize(s1["business_address"], address_steps)
    return s1

def fit_vectorizers(pool_path, sample_size=1_000_000, seed=0):
    table = pq.read_table(pool_path, columns=["name_norm", "addr_norm"])
    rows  = np.sort(np.random.default_rng(seed).choice(table.num_rows, min(sample_size, table.num_rows), replace=False))
    sample = table.take(rows).to_pandas()
    return fit_tfidf(sample["name_norm"]), fit_tfidf(sample["addr_norm"])

def pair_features(candidates, queries, pool_path, vectorizers, chunk_entities=10_000, log=print):
    t0   = time.time()
    rows = np.sort(candidates["pool_row"].unique())
    text = pq.read_table(pool_path, columns=["entity_id","name_norm","addr_norm","name_foreign"]).take(rows).to_pandas()
    text = text.rename(columns={"entity_id":"candidate_id","name_norm":"name_c","addr_norm":"addr_c","name_foreign":"name_c_foreign"})
    q    = queries[["entity_id","name_norm","addr_norm"]].rename(
           columns={"entity_id":"source1_entity_id","name_norm":"name_s1","addr_norm":"addr_s1"})
    parts = []
    for first in range(0, candidates["s1_row"].max() + 1, chunk_entities):
        chunk = candidates[(candidates["s1_row"] >= first) & (candidates["s1_row"] < first + chunk_entities)]
        if chunk.empty: continue
        chunk = (chunk[["source1_entity_id","candidate_id","s1_row","score_all","score_name","score_addr"]]
                 .merge(q, on="source1_entity_id").merge(text, on="candidate_id"))
        chunk = add_features(chunk, *(vectorizers or (None, None)))
        cols  = FEATURE_COLUMNS if vectorizers else FAST_FEATURE_COLUMNS
        parts.append(chunk[["source1_entity_id","candidate_id"] + cols])
        gc.collect()
    log(f"  features for {sum(len(p) for p in parts):,} pairs ({time.time()-t0:.0f}s)")
    return pd.concat(parts, ignore_index=True)

def label_pairs(pairs, truth_tables):
    true_keys = set()
    for t in truth_tables:
        for s1_id, ids in zip(t["source1_entity_id"], t["matched_entity_ids"]):
            true_keys.update((s1_id, m) for m in ids.split(",") if m)
    return ([(a,b) in true_keys for a,b in zip(pairs["source1_entity_id"], pairs["candidate_id"])])

def run_blocking_and_features(queries, split, name_steps, address_steps, top_k=50, max_df=1000, use_tfidf=True, log=print):
    pool_path  = normalized_pool(split, name_steps, address_steps, log=log)
    index      = build_index(queries, pool_path, max_df=max_df, chunk_rows=250_000, log=log)
    candidates = generate_candidates(queries, index, top_k=top_k, ranking="all")
    del index; gc.collect()
    log(f"  {len(candidates):,} candidates")
    vectorizers = fit_vectorizers(pool_path) if use_tfidf else None
    return candidates, pair_features(candidates, queries, pool_path, vectorizers, log=log), vectorizers

print("✅ Pipeline utilities defined.")\
"""))

# ── SECTION 4: CACHE ──────────────────────────────────────────────────────────
cells.append(cell("---\n## 💾 Section 4 — Build Parquet cache\n\nConverts TSV files to Parquet for ~20× faster reads. Run **once**.", "markdown"))
cells.append(cell("""\
import pandas as pd, time

def build_cache(split):
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    read   = lambda name: pd.read_csv(folder / name, sep="\\t", keep_default_na=False)
    pool_p = CACHE_DIR / f"{split}_pool.parquet"
    if not pool_p.exists():
        t = time.time()
        pool = pd.concat([read(f"{split}_source2.tsv"), read(f"{split}_source3.tsv")], ignore_index=True)
        pool.to_parquet(pool_p, row_group_size=1_000_000, index=False); del pool
        print(f"  {split} pool cached ({time.time()-t:.0f}s)")
    else:
        print(f"  {split} pool: already cached")
    s1_p = CACHE_DIR / f"{split}_source1.parquet"
    if not s1_p.exists():
        t = time.time()
        read(f"{split}_source1.tsv").to_parquet(s1_p, index=False)
        print(f"  {split} source1 cached ({time.time()-t:.0f}s)")
    else:
        print(f"  {split} source1: already cached")

print("Building cache for train …"); build_cache("train")
print("Building cache for test  …"); build_cache("test")
print("✅ Cache ready.")\
"""))

# ── SECTION 5: TRAINING ───────────────────────────────────────────────────────
cells.append(cell("---\n## 🤖 Section 5 — Training (E4 configuration)\n\n**E4**: transliteration + K=100 + 32 fast features + 40 k training entities + HistGradientBoosting (400 iters).\n\n⚠ Blocking + features step takes ~15–45 min depending on dataset size.", "markdown"))

cells.append(cell("""\
import numpy as np

TRANSLIT_NAME    = ("translit",) + NAME_STEPS
TRANSLIT_ADDRESS = ("translit",) + ADDRESS_STEPS

E4_CONFIG = dict(
    name_steps    = TRANSLIT_NAME,
    address_steps = TRANSLIT_ADDRESS,
    extra_train   = 20_000,
    top_k         = 100,
    use_tfidf     = False,
)
HGB_PARAMS = dict(max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
                   min_samples_leaf=50, l2_regularization=1.0,
                   early_stopping=False, random_state=0)
THRESHOLDS = np.round(np.arange(0.50, 0.91, 0.025), 3)

print("✅ E4 config ready.")\
"""))

cells.append(cell("""\
import pandas as pd, import_pyarrow_parquet_as_pq

gt    = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\\t", keep_default_na=False)
split = make_validation_split(gt)

tuning_ids     = split.loc[split["split"] == "train", "source1_entity_id"].sample(20_000, random_state=7)
validation_ids = split.loc[split["small_validation"], "source1_entity_id"]

truth = {
    "tuning":     gt[gt["source1_entity_id"].isin(tuning_ids)].reset_index(drop=True),
    "validation": gt[gt["source1_entity_id"].isin(validation_ids)].reset_index(drop=True),
}
query_set = pd.concat([pd.DataFrame({"entity_id": tuning_ids, "query_set": "tuning"}),
                        pd.DataFrame({"entity_id": validation_ids, "query_set": "validation"})])
s1_base = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
s1      = s1_base.merge(query_set, on="entity_id").reset_index(drop=True)

if E4_CONFIG["extra_train"]:
    pool_ids  = split.loc[(split["split"]=="train") & ~split["source1_entity_id"].isin(s1["entity_id"]), "source1_entity_id"]
    extra_ids = pool_ids.sample(E4_CONFIG["extra_train"], random_state=11)
    extra     = s1_base[s1_base["entity_id"].isin(extra_ids)].assign(query_set="tuning")
    s1        = pd.concat([s1, extra], ignore_index=True)
    truth["tuning"] = pd.concat([truth["tuning"], gt[gt["source1_entity_id"].isin(extra_ids)]], ignore_index=True)

print(f"tuning: {(s1['query_set']=='tuning').sum():,}  validation: {(s1['query_set']=='validation').sum():,}")\
""".replace("import import_pyarrow_parquet_as_pq\n", "")))

cells.append(cell("""\
import time, gc
_t0 = time.time()
log = lambda msg: print(f"[{time.time()-_t0:5.0f}s] {msg}", flush=True)

queries = prepare_queries(s1, E4_CONFIG["name_steps"], E4_CONFIG["address_steps"])
candidates, pairs, vectorizers = run_blocking_and_features(
    queries, "train", E4_CONFIG["name_steps"], E4_CONFIG["address_steps"],
    top_k=E4_CONFIG["top_k"], use_tfidf=E4_CONFIG["use_tfidf"], log=log)

print(f"\\n✅ pairs shape: {pairs.shape}")\
"""))

cells.append(cell("""\
import numpy as np

features = FAST_FEATURE_COLUMNS if not E4_CONFIG["use_tfidf"] else FEATURE_COLUMNS
pairs = pairs.merge(queries[["entity_id","query_set"]], left_on="source1_entity_id", right_on="entity_id").drop(columns="entity_id")
pairs["label"] = label_pairs(pairs, list(truth.values()))
pairs["label"] = [int(v) for v in pairs["label"]]

for s in ["tuning", "validation"]:
    t      = truth[s]
    n_true = t["matched_entity_ids"].map(lambda x: len(x.split(",")) if x else 0).sum()
    recall = pairs.loc[pairs["query_set"]==s, "label"].sum() / n_true
    print(f"  {s} candidate recall: {recall:.4f}")

del candidates; gc.collect()
print("✅ Labels done.")\
"""))

cells.append(cell("""\
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

train_df = pairs[pairs["query_set"]=="tuning"].reset_index(drop=True)
valid_df  = pairs[pairs["query_set"]=="validation"].reset_index(drop=True)
del pairs; gc.collect()

X = train_df[features].astype("float32")
y = np.array(train_df["label"], dtype="int8")
print(f"Training {len(train_df):,} pairs ({y.sum():,} positives) | {len(features)} features")

oof = np.zeros(len(train_df))
for i, (fit_idx, pred_idx) in enumerate(GroupKFold(n_splits=3).split(X, y, groups=train_df["source1_entity_id"])):
    m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X.iloc[fit_idx], y[fit_idx])
    oof[pred_idx] = m.predict_proba(X.iloc[pred_idx])[:, 1]
    print(f"  fold {i+1} done")

def macro_score(df, probs, thr, truth_table):
    chosen = global_threshold(df.assign(probability=probs), thr)
    return score_report(to_submission(resolve_conflicts(chosen)), truth_table)["macro_F0.5"]

scores = [macro_score(train_df, oof, t, truth["tuning"]) for t in THRESHOLDS]
best_threshold = float(THRESHOLDS[int(np.argmax(scores))])
print(f"\\n  Best threshold: {best_threshold}  (tuning OOF F0.5 = {max(scores):.4f})")

model = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
print("✅ Final model trained.")\
"""))

cells.append(cell("""\
p_valid = model.predict_proba(valid_df[features].astype("float32"))[:, 1]
rep = score_report(
    to_submission(resolve_conflicts(global_threshold(valid_df.assign(probability=p_valid), best_threshold))),
    truth["validation"])
print(f"{'='*55}")
print(f"  Validation macro F0.5 : {rep['macro_F0.5']:.4f}")
print(f"  Pair precision        : {rep['pair precision']:.4f}")
print(f"  Pair recall           : {rep['pair recall']:.4f}")
print(f"{'='*55}")\
"""))

cells.append(cell("""\
import joblib

artifacts = {
    "model": model, "threshold": best_threshold,
    "vectorizers": vectorizers, "features": features,
    "name_steps": E4_CONFIG["name_steps"], "address_steps": E4_CONFIG["address_steps"],
    "top_k": E4_CONFIG["top_k"], "max_df": 1000,
}
model_path = MODELS_DIR / "E4.joblib"
joblib.dump(artifacts, model_path)
print(f"✅ Model saved to: {model_path}")\
"""))

# ── SECTION 6: PREDICTION ─────────────────────────────────────────────────────
cells.append(cell("---\n## 🔮 Section 6 — Test set prediction\n\nProcesses one **country at a time** with checkpointing.  \n⚠ Full test set: ~2.5 h on Kaggle (16 GB RAM). Colab free tier may be slower.", "markdown"))

cells.append(cell("""\
import joblib

model_path = MODELS_DIR / "E4.joblib"
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Run Section 5 first.")
artifacts = joblib.load(model_path)
print(f"✅ Artifacts loaded | threshold={artifacts['threshold']} | top_k={artifacts['top_k']}")\
"""))

cells.append(cell("""\
import gc, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq

KEEP_PROBABILITY = 0.30

def score_country_inline(queries, pool_path, n_pool, artifacts, country,
                          candidate_file, chunk_rows=250_000, yield_every=5_000, log=print):
    t0 = time.time()
    q  = queries[queries["country"] == country].reset_index(drop=True)
    pool = pq.read_table(pool_path,
        columns=["entity_id","country","name_norm","addr_norm","name_foreign"],
        filters=[("country","=",country)])
    index = build_index_hashed(q, (b.to_pandas() for b in pool.to_batches(max_chunksize=chunk_rows)),
                                n_pool=n_pool, max_df=artifacts["max_df"], log=log)
    q_text = q[["entity_id","name_norm","addr_norm"]].rename(
        columns={"entity_id":"source1_entity_id","name_norm":"name_s1","addr_norm":"addr_s1"})
    scored, written = [], []
    n_done = 0
    for cands in iter_candidates(q, index, top_k=artifacts["top_k"], batch_size=1_000, yield_every=yield_every):
        text = pool.take(cands["pool_row"].to_numpy()).to_pandas()
        p    = cands[["source1_entity_id","s1_row","score_all","score_name","score_addr"]].reset_index(drop=True)
        p["candidate_id"]  = text["entity_id"].to_numpy()
        p["name_c"]        = text["name_norm"].to_numpy()
        p["addr_c"]        = text["addr_norm"].to_numpy()
        p["name_c_foreign"]= text["name_foreign"].to_numpy()
        p = p.merge(q_text, on="source1_entity_id")
        p = add_features(p, *(artifacts["vectorizers"] or (None, None)))
        p["probability"] = artifacts["model"].predict_proba(p[artifacts["features"]].astype("float32"))[:, 1]
        lists = p.groupby("source1_entity_id")["candidate_id"].agg(",".join)
        candidate_file.write("".join(f"{sid}\\t{ids}\\n" for sid, ids in lists.items()))
        written.append(lists.index.to_numpy())
        scored.append(p.loc[p["probability"] >= KEEP_PROBABILITY, ["source1_entity_id","candidate_id","probability"]])
        n_done += p["source1_entity_id"].nunique()
        log(f"  {country}: {n_done:,}/{len(q):,} entities scored ({time.time()-t0:.0f}s)")
        del p, text; gc.collect()
    del index, pool; gc.collect()
    return (pd.concat(scored, ignore_index=True),
            np.concatenate(written) if written else np.array([], dtype=object))

# ── Main loop ────────────────────────────────────────────────────────────────
t0  = time.time()
log = lambda msg: print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

s1_test   = pd.read_parquet(CACHE_DIR / "test_source1.parquet")
queries   = prepare_queries(s1_test, artifacts["name_steps"], artifacts["address_steps"])
pool_path = normalized_pool("test", artifacts["name_steps"], artifacts["address_steps"], log=log)
n_pool    = pq.ParquetFile(pool_path).metadata.num_rows

PARTS_DIR = CACHE_DIR / "predict_parts_test"
PARTS_DIR.mkdir(parents=True, exist_ok=True)

scored_all, written_all = [], []
for country in sorted(queries["country"].unique()):
    safe  = "".join(ch if ch.isalnum() else "_" for ch in country)
    done  = PARTS_DIR / f"{safe}.done"
    sc_f  = PARTS_DIR / f"{safe}_scored.parquet"
    id_f  = PARTS_DIR / f"{safe}_ids.parquet"
    c_f   = PARTS_DIR / f"{safe}_candidates.tsv"
    if done.exists():
        log(f"  {country}: checkpoint found, skipping")
    else:
        with open(c_f, "w", encoding="utf-8", newline="\\n") as cf:
            p_c, id_c = score_country_inline(queries, pool_path, n_pool, artifacts, country, cf, log=log)
        p_c.to_parquet(sc_f, index=False)
        pd.DataFrame({"source1_entity_id": id_c}).to_parquet(id_f, index=False)
        done.touch(); del p_c, id_c; gc.collect()
    scored_all.append(pd.read_parquet(sc_f))
    written_all.append(pd.read_parquet(id_f)["source1_entity_id"].to_numpy())

log("All countries done — assembling outputs …")\
"""))

cells.append(cell("""\
import numpy as np, pandas as pd

# candidate_pairs.tsv
cand_out = OUTPUT_DIR / "candidate_pairs.tsv"
with open(cand_out, "w", encoding="utf-8", newline="\\n") as cf:
    cf.write("source1_entity_id\\tcandidate_entity_ids\\n")
    for part in sorted(PARTS_DIR.glob("*_candidates.tsv")):
        with open(part, encoding="utf-8", newline="") as f:
            for block in iter(lambda: f.read(1<<24), ""): cf.write(block)
    without = s1_test.loc[~s1_test["entity_id"].isin(np.concatenate(written_all)), "entity_id"]
    cf.write("".join(f"{sid}\\t\\n" for sid in without))
log(f"candidate_pairs.tsv written ({len(without):,} entities without candidates)")

# matching_results.tsv
scored   = pd.concat(scored_all, ignore_index=True)
matches  = resolve_conflicts(global_threshold(scored, artifacts["threshold"]))
ml       = matches.groupby("source1_entity_id")["candidate_id"].agg(",".join)
matching = pd.DataFrame({"source1_entity_id": s1_test["entity_id"],
                          "matched_entity_ids": s1_test["entity_id"].map(ml).fillna("").to_numpy()})

match_out = OUTPUT_DIR / "matching_results.tsv"
matching[["source1_entity_id","matched_entity_ids"]].to_csv(
    match_out, sep="\\t", index=False, encoding="utf-8", lineterminator="\\n")

n_with = (matching["matched_entity_ids"] != "").sum()
print(f"\\n✅ {match_out.name}: {n_with:,}/{len(s1_test):,} entities with matches")
print(f"   {cand_out.name}: written")\
"""))

# ── SECTION 7: DOWNLOAD ────────────────────────────────────────────────────────
cells.append(cell("---\n## 📥 Section 7 — Preview & download outputs", "markdown"))
cells.append(cell("""\
import pandas as pd
print("=== matching_results.tsv (first 10 rows) ===")
display(pd.read_csv(OUTPUT_DIR / "matching_results.tsv", sep="\\t", keep_default_na=False).head(10))\
"""))
cells.append(cell("""\
from google.colab import files
for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
    p = OUTPUT_DIR / fname
    if p.exists(): files.download(str(p))
    else: print(f"⚠️  {fname} not found — run prediction section first.")\
"""))

# ── SECTION 8: VALIDATION (OPTIONAL) ─────────────────────────────────────────
cells.append(cell("---\n## 🧪 Section 8 (Optional) — Full validation scoring\n\nReproduce logged metrics on the full 441 k validation split.", "markdown"))
cells.append(cell("""\
import gc, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq

t0  = time.time()
log = lambda msg: print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

gt_val   = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\\t", keep_default_na=False)
split_df = make_validation_split(gt_val)
val_ids  = split_df.loc[split_df["split"] == "validation", "source1_entity_id"]

s1_val   = pd.read_parquet(CACHE_DIR / "train_source1.parquet")
s1_val   = s1_val[s1_val["entity_id"].isin(val_ids)].reset_index(drop=True)
queries_v= prepare_queries(s1_val, artifacts["name_steps"], artifacts["address_steps"])
pool_p   = normalized_pool("train", artifacts["name_steps"], artifacts["address_steps"], log=log)
n_pool_v = pq.ParquetFile(pool_p).metadata.num_rows

PARTS_V  = CACHE_DIR / "predict_parts_validation"
PARTS_V.mkdir(parents=True, exist_ok=True)

scored_v, written_v = [], []
for country in sorted(queries_v["country"].unique()):
    safe = "".join(ch if ch.isalnum() else "_" for ch in country)
    done = PARTS_V / f"{safe}.done"; sc_f = PARTS_V / f"{safe}_scored.parquet"
    id_f = PARTS_V / f"{safe}_ids.parquet"; c_f = PARTS_V / f"{safe}_candidates.tsv"
    if done.exists():
        log(f"  {country}: checkpoint, skipping")
    else:
        with open(c_f, "w", encoding="utf-8", newline="\\n") as cf:
            p_c, id_c = score_country_inline(queries_v, pool_p, n_pool_v, artifacts, country, cf, log=log)
        p_c.to_parquet(sc_f, index=False)
        pd.DataFrame({"source1_entity_id": id_c}).to_parquet(id_f, index=False)
        done.touch(); del p_c, id_c; gc.collect()
    scored_v.append(pd.read_parquet(sc_f))
    written_v.append(pd.read_parquet(id_f)["source1_entity_id"].to_numpy())

scored_v = pd.concat(scored_v, ignore_index=True)
matches_v= resolve_conflicts(global_threshold(scored_v, artifacts["threshold"]))
ml_v     = matches_v.groupby("source1_entity_id")["candidate_id"].agg(",".join)
matching_v = pd.DataFrame({"source1_entity_id": s1_val["entity_id"],
                             "matched_entity_ids": s1_val["entity_id"].map(ml_v).fillna("").to_numpy()})
truth_full = gt_val[gt_val["source1_entity_id"].isin(val_ids)]
rep = score_report(matching_v, truth_full)
print(f"{'='*55}")
print(f"  Full validation macro F0.5 : {rep['macro_F0.5']:.4f}")
print(f"  Pair precision             : {rep['pair precision']:.4f}")
print(f"  Pair recall                : {rep['pair recall']:.4f}")
print(f"{'='*55}")\
"""))

# ── BUILD NOTEBOOK JSON ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "collapsed_sections": []},
    },
    "cells": cells,
}

out_path = "/Users/giriprasad/Desktop/Amazon/ml-_challange/Amazon_ML_Challenge_Colab.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✅ Notebook written to: {out_path}")
