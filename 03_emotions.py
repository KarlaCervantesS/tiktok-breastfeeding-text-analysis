"""
3. Emotions of the comments with RoBERTuito (Ekman's six basic emotions).

    Model  pysentimiento/robertuito-emotion-analysis (config.EMOTION_MODEL), which
           assigns each comment ONE of: anger, disgust, fear, joy, sadness, surprise,
           or "others" (reported here as "No emotion").

If 02_sentiment.py was run before, the sentiment of the comments with each emotion
is also reported (consistency check between the two models).

Outputs (folder <OUTPUT_DIR>/emotions/)
    emotion_predictions.xlsx   every comment with its emotion and probabilities
    emotions.pdf/.png          bar chart of the comments with an emotion
    emotion_tables.xlsx/.tex
    summary.txt
"""
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from models import classify
from utils import Log, load_comments, output_dir, package_versions

OUT = output_dir("emotions")
SENTIMENT_FILE = os.path.join(config.OUTPUT_DIR, "sentiment", "sentiment_predictions.xlsx")
say = Log()

EKMAN = ["Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust"]
NONE = "No emotion"
CATEGORIES = EKMAN + [NONE]
LABELS = {"joy": "Joy", "sadness": "Sadness", "anger": "Anger", "fear": "Fear",
          "surprise": "Surprise", "disgust": "Disgust", "others": NONE}
COLORS = {"Joy": "#eda100", "Sadness": "#2a78d6", "Anger": "#e34948",
          "Fear": "#6250d6", "Surprise": "#eb6834", "Disgust": "#1baf7a"}

# =====================================================================
# 1. Classification
# =====================================================================
df, n_file = load_comments()
n = len(df)
say("EMOTIONS (RoBERTuito, Ekman's basic emotions)")
say("=" * 72)
say(f"Date: {datetime.now():%Y-%m-%d %H:%M}")
say(f"Input: {config.INPUT_FILE} ({n_file} rows; {n} with text)")

p, labels, n_trunc = classify(df[config.TEXT_COLUMN], config.EMOTION_MODEL)
for j, lab in enumerate(labels):
    df[f"p_{LABELS[lab].lower().replace(' ', '_')}"] = p[:, j]
df["emotion"] = [LABELS[labels[k]] for k in p.argmax(axis=1)]
df["confidence"] = p.max(axis=1)
say(f"Model: {config.EMOTION_MODEL} ({n_trunc} comments truncated at "
    f"{config.MAX_LENGTH} tokens)\n")

# =====================================================================
# 2. Distribution
# =====================================================================
counts = df["emotion"].value_counts().reindex(CATEGORIES, fill_value=0)
n_emo = int((df["emotion"] != NONE).sum())
table1 = pd.DataFrame({
    "n": counts,
    "% of all comments": (100 * counts / n).round(1),
    "% of comments with an emotion": (100 * counts / max(n_emo, 1)).round(1),
    "Mean confidence": df.groupby("emotion")["confidence"].mean().reindex(CATEGORIES).round(2)})
table1.loc[NONE, "% of comments with an emotion"] = np.nan
table1.index.name = "Emotion"
say(f"TABLE 1. Emotion of each comment (n = {n}; with an emotion: {n_emo}, "
    f"{100 * n_emo / n:.1f}%)")
say(table1.to_string())
say()
tables = {"emotions": table1}

# Bar chart: comments with an emotion (emotions never assigned are not drawn)
pct = table1.loc[EKMAN, "% of comments with an emotion"].sort_values(ascending=False)
pct = pct[counts.reindex(pct.index) > 0]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(pct))
ax.bar(x, pct.values, 0.6, color=[COLORS[e] for e in pct.index], zorder=2)
for xp, v, k in zip(x, pct.values, counts.reindex(pct.index).values):
    ax.text(xp, v + 0.8, f"{v:.1f}%\n(n = {k:,})", ha="center", va="bottom",
            fontsize=8, color="#333333")
ax.set_xticks(x, pct.index)
ax.set_ylabel("Comments with an emotion (%)")
ax.set_ylim(0, max(pct.max(), 1) * 1.25)
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"emotions.{ext}"), dpi=300)
plt.close(fig)

# =====================================================================
# 3. Emotion x sentiment (if 02_sentiment.py was run)
# =====================================================================
if os.path.exists(SENTIMENT_FILE):
    sent = pd.read_excel(SENTIMENT_FILE)[[config.ID_COLUMN, "sentiment"]]
    df = df.merge(sent, on=config.ID_COLUMN, how="left")
    classes = ["Negative", "Neutral", "Positive"]
    ct = pd.crosstab(df["emotion"], df["sentiment"]).reindex(index=CATEGORIES,
                                                            columns=classes, fill_value=0)
    table2 = (100 * ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)).round(1)
    table2.insert(0, "n", ct.sum(axis=1))
    table2.index.name, table2.columns.name = "Emotion", None
    say("TABLE 2. Sentiment of the comments with each emotion (% of the row)")
    say(table2.to_string())
    say()
    tables["emotion_x_sentiment"] = table2
else:
    say("(Run 02_sentiment.py first to obtain the emotion x sentiment table.)\n")

# =====================================================================
# 4. Save
# =====================================================================
keep = [config.ID_COLUMN, config.TEXT_COLUMN, "emotion", "confidence"] + \
       [c for c in df.columns if c.startswith("p_")] + \
       (["sentiment"] if "sentiment" in df.columns else [])
df[keep].to_excel(os.path.join(OUT, "emotion_predictions.xlsx"), index=False)
with pd.ExcelWriter(os.path.join(OUT, "emotion_tables.xlsx")) as w:
    for name, t in tables.items():
        t.to_excel(w, sheet_name=name[:31])
with open(os.path.join(OUT, "emotion_tables.tex"), "w", encoding="utf-8") as f:
    for name, t in tables.items():
        f.write(f"% {name}\n" + t.to_latex(float_format="%.1f", escape=True, na_rep="--") + "\n")

say("Software: " + package_versions(["torch", "transformers", "pysentimiento"]))
say(f"Results saved in: {OUT}/")
say.save(os.path.join(OUT, "summary.txt"))
