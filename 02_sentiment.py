"""
2. Sentiment of the comments with RoBERTuito and (optional) validation against
human labels.

    Model  pysentimiento/robertuito-sentiment-analysis (config.SENTIMENT_MODEL);
           each comment is assigned the class with the highest probability
           (Negative, Neutral or Positive).

Validation (only if config.VALIDATION_FILE is set): the human-labeled comments are
located in the input table by id or, if there are no ids, by text (exact match,
then the most similar text above config.FUZZY_MIN_SCORE and clearly better than
the second best). Metrics: precision, recall and F1 per class, macro and weighted
averages, confusion matrix, accuracy with 95% Wilson confidence interval, Cohen's
kappa, and number of opposite labels (Negative vs Positive).

Outputs (folder <OUTPUT_DIR>/sentiment/)
    sentiment_predictions.xlsx   every comment with its class and probabilities
    sentiment_distribution.pdf/.png
    validation_matches.xlsx      (if validation) human-labeled comments and labels
    sentiment_tables.xlsx/.tex   distribution and validation tables
    summary.txt
"""
import difflib
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support)

import config
from models import classify
from utils import Log, load_comments, output_dir, package_versions, read_table

OUT = output_dir("sentiment")
say = Log()
CLASSES = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#e34948", "Neutral": "#8a9096", "Positive": "#1baf7a"}


def norm_label(x):
    """'negative', 'NEG', 'Negative ' -> 'Negative' (None if not recognized)."""
    s = str(x).strip().lower()
    return next((c for c in CLASSES if s.startswith(c[:3].lower())), None)


def norm_text(x):
    """Matching key: lower case, no punctuation, single spaces."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(x).lower())).strip()


# =====================================================================
# 1. Classification
# =====================================================================
df, n_file = load_comments()
n = len(df)
say("SENTIMENT CLASSIFICATION (RoBERTuito)")
say("=" * 72)
say(f"Date: {datetime.now():%Y-%m-%d %H:%M}")
say(f"Input: {config.INPUT_FILE} ({n_file} rows; {n} with text)")

p, labels, n_trunc = classify(df[config.TEXT_COLUMN], config.SENTIMENT_MODEL)
order = [norm_label(l) for l in labels]                 # model order -> CLASSES order
p = p[:, [order.index(c) for c in CLASSES]]
for j, c in enumerate(CLASSES):
    df[f"p_{c.lower()}"] = p[:, j]
df["sentiment"] = [CLASSES[k] for k in p.argmax(axis=1)]
df["confidence"] = p.max(axis=1)
say(f"Model: {config.SENTIMENT_MODEL} ({n_trunc} comments truncated at "
    f"{config.MAX_LENGTH} tokens)\n")

# =====================================================================
# 2. Distribution
# =====================================================================
counts = df["sentiment"].value_counts().reindex(CLASSES, fill_value=0)
pct = 100 * counts / n
table1 = pd.DataFrame({"n": counts, "%": pct.round(1),
                       "Mean confidence": df.groupby("sentiment")["confidence"]
                       .mean().reindex(CLASSES).round(2)})
table1.index.name = "Sentiment"
say(f"TABLE 1. Distribution of sentiment (n = {n})")
say(table1.to_string())
say()
tables = {"distribution": (table1, True)}

fig, ax = plt.subplots(figsize=(5.5, 4))
x = np.arange(len(CLASSES))
ax.bar(x, pct.values, 0.6, color=[COLORS[c] for c in CLASSES], zorder=2)
for xp, v, k in zip(x, pct.values, counts.values):
    ax.text(xp, v + 0.8, f"{v:.1f}%\n(n = {k:,})", ha="center", va="bottom",
            fontsize=9, color="#333333")
ax.set_xticks(x, CLASSES)
ax.set_ylabel("Comments (%)")
ax.set_ylim(0, pct.max() * 1.25)
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"sentiment_distribution.{ext}"), dpi=300)
plt.close(fig)


# =====================================================================
# 3. Validation against human labels (optional)
# =====================================================================
def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    p_ = k / n
    centre = (p_ + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p_ * (1 - p_) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def match_by_text(val):
    """Row of the input table for each validation text (exact, then similar)."""
    pos = {}
    for i, k in df[config.CORPUS_MATCH_COLUMN].apply(norm_text).items():
        pos.setdefault(k, i)
    keys = list(pos)
    rows, methods = [], []
    for text in val[config.VALIDATION_TEXT_COLUMN]:
        key = norm_text(text)
        if key in pos:
            rows.append(pos[key]); methods.append("exact")
            continue
        best = difflib.get_close_matches(key, keys, n=2, cutoff=0.5)
        row, method = None, "not found"
        if best:
            s1 = difflib.SequenceMatcher(None, key, best[0]).ratio()
            s2 = difflib.SequenceMatcher(None, key, best[1]).ratio() if len(best) > 1 else 0
            if s1 >= config.FUZZY_MIN_SCORE and s1 - s2 >= config.FUZZY_MARGIN:
                row, method = pos[best[0]], "similar"
            elif s1 >= config.FUZZY_MIN_SCORE:
                method = "ambiguous (not used)"
        rows.append(row); methods.append(method)
    return rows, methods


if config.VALIDATION_FILE:
    val = read_table(config.VALIDATION_FILE, config.VALIDATION_SHEET)
    val["human"] = val[config.VALIDATION_LABEL_COLUMN].apply(norm_label)
    if config.VALIDATION_ID_COLUMN and config.VALIDATION_ID_COLUMN in val.columns:
        pos_by_id = {v: i for i, v in df[config.ID_COLUMN].items()}
        val["row"] = val[config.VALIDATION_ID_COLUMN].map(pos_by_id)
        val["match_method"] = np.where(val["row"].notna(), "id", "not found")
    else:
        val["row"], val["match_method"] = match_by_text(val)
    say("VALIDATION: matching of the human-labeled comments")
    say(val["match_method"].value_counts().to_string())
    say()

    val = val[val["row"].notna() & val["human"].notna()].copy()
    rows = val["row"].astype(int)
    val["comment_id"] = df.loc[rows, config.ID_COLUMN].values
    val["model"] = df.loc[rows, "sentiment"].values
    y_ref, y_pred = val["human"], val["model"]
    val.drop(columns="row").to_excel(os.path.join(OUT, "validation_matches.xlsx"),
                                     index=False)

    pr, rc, f1, sup = precision_recall_fscore_support(y_ref, y_pred, labels=CLASSES,
                                                      zero_division=0)
    table2 = pd.DataFrame({"Precision": pr, "Recall": rc, "F1": f1, "Support": sup},
                          index=CLASSES)
    for avg, name in [("macro", "Macro average"), ("weighted", "Weighted average")]:
        pa, ra, fa, _ = precision_recall_fscore_support(y_ref, y_pred, labels=CLASSES,
                                                        average=avg, zero_division=0)
        table2.loc[name] = [pa, ra, fa, len(y_ref)]
    table2["Support"] = table2["Support"].astype(int)
    table2 = table2.round(2)
    table3 = pd.DataFrame(confusion_matrix(y_ref, y_pred, labels=CLASSES),
                          index=[f"Human: {c}" for c in CLASSES],
                          columns=[f"Model: {c}" for c in CLASSES])
    k_ok, n_val = int((y_ref == y_pred).sum()), len(val)
    lo, hi = wilson_ci(k_ok, n_val)
    opposite = int((((y_ref == "Negative") & (y_pred == "Positive"))
                    | ((y_ref == "Positive") & (y_pred == "Negative"))).sum())
    table4 = pd.DataFrame({
        "Measure": ["Labeled comments", "Correct", "Accuracy", "95% CI (Wilson)",
                    "Macro-F1", "Cohen's kappa", "Opposite labels (Negative vs Positive)"],
        "Value": [str(n_val), str(k_ok), f"{k_ok / n_val:.2f}", f"{lo:.2f}-{hi:.2f}",
                  f"{table2.loc['Macro average', 'F1']:.2f}",
                  f"{cohen_kappa_score(y_ref, y_pred, labels=CLASSES):.2f}", str(opposite)]})
    say(f"TABLE 2. Model vs human labels (n = {n_val})\n{table2.to_string()}\n")
    say(f"TABLE 3. Confusion matrix (rows = human, columns = model)\n{table3.to_string()}\n")
    say(f"TABLE 4. Agreement\n{table4.to_string(index=False)}\n")
    tables.update({"vs_human": (table2, True), "confusion": (table3, True),
                   "agreement": (table4, False)})

# =====================================================================
# 4. Save
# =====================================================================
keep = [config.ID_COLUMN, config.TEXT_COLUMN, "sentiment", "confidence"] + \
       [f"p_{c.lower()}" for c in CLASSES]
df[keep].to_excel(os.path.join(OUT, "sentiment_predictions.xlsx"), index=False)
with pd.ExcelWriter(os.path.join(OUT, "sentiment_tables.xlsx")) as w:
    for name, (t, idx) in tables.items():
        t.to_excel(w, sheet_name=name, index=idx)
with open(os.path.join(OUT, "sentiment_tables.tex"), "w", encoding="utf-8") as f:
    for name, (t, idx) in tables.items():
        f.write(f"% {name}\n" + t.to_latex(index=idx, float_format="%.2f", escape=True) + "\n")

say("Software: " + package_versions(["torch", "transformers", "pysentimiento", "scikit-learn"]))
say(f"Results saved in: {OUT}/")
say.save(os.path.join(OUT, "summary.txt"))
