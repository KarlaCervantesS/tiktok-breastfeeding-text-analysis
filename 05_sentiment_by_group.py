"""
5. Sentiment of the comments by group of topics (e.g. themes) and by topic.

Requires the outputs of 02_sentiment.py and 04_topics.py, and a file
(config.TOPIC_GROUPS_FILE) with one row per topic and three columns:
    topic   topic number as in 04_topics.py (e.g. 1 for T1)
    label   short label shown in the figure
    group   name of the group (theme) the topic belongs to
Groups are drawn in the order in which they first appear in the file; topics not
listed are ignored. See examples/topic_groups_example.csv.

Figure: one stacked horizontal bar per group (bold), followed by its topics, plus
reference bars for all comments and for the outliers.

Outputs (folder <OUTPUT_DIR>/topics/)
    sentiment_by_group.pdf/.png, sentiment_by_group.xlsx/.tex
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import output_dir, read_table

OUT = output_dir("topics")
TOPICS_FILE = os.path.join(config.OUTPUT_DIR, "topics", "topics.xlsx")
SENTIMENT_FILE = os.path.join(config.OUTPUT_DIR, "sentiment", "sentiment_predictions.xlsx")

SHOW_TOPICS = True        # False = one bar per group only
SHOW_ALL = True           # reference bar with all comments
SHOW_OUTLIERS = True      # bar with the comments not assigned to any topic
MIN_LABEL = 8             # write a percentage inside a segment only if >= this value
CLASSES = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#e34948", "Neutral": "#8a9096", "Positive": "#1baf7a"}

# =====================================================================
# 1. Data
# =====================================================================
df = pd.read_excel(TOPICS_FILE, sheet_name="comments")
if "sentiment" not in df.columns:
    sent = pd.read_excel(SENTIMENT_FILE)[[config.ID_COLUMN, "sentiment"]]
    df = df.merge(sent, on=config.ID_COLUMN, how="left")
df = df.dropna(subset=["sentiment"])

groups = read_table(config.TOPIC_GROUPS_FILE)
groups["topic"] = groups["topic"].astype(int)
missing = sorted(set(df["topic"]) - set(groups["topic"]) - {-1})
if missing:
    print(f"WARNING: topics not listed in {config.TOPIC_GROUPS_FILE} (not drawn): {missing}")


def row(sub, label, kind):
    counts = sub["sentiment"].value_counts().reindex(CLASSES, fill_value=0)
    out = {"label": label, "kind": kind, "n": int(counts.sum())}
    out.update((100 * counts / max(counts.sum(), 1)).round(1).to_dict())
    return out


rows = [row(df, "All comments", "all")] if SHOW_ALL else []
for group in groups["group"].drop_duplicates():
    g = groups[groups["group"] == group]
    rows.append(row(df[df["topic"].isin(g["topic"])], group, "group"))
    if SHOW_TOPICS:
        for _, r in g.iterrows():
            rows.append(row(df[df["topic"] == r["topic"]], r["label"], "topic"))
if SHOW_OUTLIERS:
    rows.append(row(df[df["topic"] == -1], "Not assigned to any topic (outliers)", "outliers"))
table = pd.DataFrame(rows)
print(table.to_string(index=False))

# =====================================================================
# 2. Figure
# =====================================================================
ypos, y = [], 0.0
for k in table["kind"]:
    if k in ("group", "outliers") and ypos:
        y += 0.5
    ypos.append(y)
    y += 1
ypos = np.array(ypos)

fig, ax = plt.subplots(figsize=(8, 0.42 * y + 1.3))
left = np.zeros(len(table))
heights = np.where(table["kind"] == "topic", 0.7, 0.85)
for c in CLASSES:
    ax.barh(ypos, table[c], left=left, height=heights, color=COLORS[c], label=c,
            edgecolor="white", linewidth=1)
    for i, (l, v) in enumerate(zip(left, table[c])):
        if v >= MIN_LABEL:
            ax.text(l + v / 2, ypos[i], f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white", weight="normal" if table.loc[i, "kind"] == "topic" else "bold")
    left += table[c].values
ax.set_yticks(ypos)
ax.set_yticklabels([f"{lab} (n = {n:,})" for lab, n in zip(table["label"], table["n"])],
                   fontsize=8)
for tick, k in zip(ax.get_yticklabels(), table["kind"]):
    if k != "topic":
        tick.set_fontweight("bold")
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Comments (%)")
ax.tick_params(axis="y", length=0)
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"sentiment_by_group.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 3. Table
# =====================================================================
out = table.drop(columns="kind").rename(columns={"label": "Group / topic"})
out.to_excel(os.path.join(OUT, "sentiment_by_group.xlsx"), index=False)
with open(os.path.join(OUT, "sentiment_by_group.tex"), "w", encoding="utf-8") as f:
    f.write(out.to_latex(index=False, float_format="%.1f", escape=True))
print(f"\nResults saved in: {OUT}/")
