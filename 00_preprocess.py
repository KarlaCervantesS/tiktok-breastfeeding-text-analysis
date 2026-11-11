"""
0. Preprocessing of the raw comments.

Steps (applied in this order; the number of comments removed in each step is
reported)
    1. Each comment receives an id = its row number in the raw file, assigned before
       any cleaning, so that every comment can be traced back to the raw data.
    2. Empty records are removed.
    3. Exact duplicates are removed (the first occurrence is kept).
    4. Comments containing user mentions (@user) are removed.
    5. Comments containing URLs are removed.
    6. The text is cleaned: lower case; hashtag symbols removed (the word is kept);
       emojis and punctuation removed; characters repeated three or more times
       reduced to two ("holaaaa" -> "holaa"); extra spaces removed.
    7. Comments left empty after cleaning (e.g. only emojis) are removed.

The cleaned comments are written to config.INPUT_FILE (columns ID_COLUMN and
TEXT_COLUMN), which is the input of the analysis scripts. Any manual review of the
comments (e.g. excluding unrelated comments or replacing personal names) should be
done on that file before running the analyses.

Outputs
    config.INPUT_FILE                         cleaned comments
    <OUTPUT_DIR>/preprocessing/summary.txt    comments removed in each step and
                                              number of words per comment
"""
import os
import re
from datetime import datetime

import pandas as pd

import config
from utils import Log, output_dir, read_table

OUT = output_dir("preprocessing")
say = Log()
steps = []


def record(name, data):
    """Keep the name of the step and the number of comments left after it."""
    steps.append((name, len(data)))


EMOJI = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
                   "\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U000024C2-\U0001F251]+")


def clean(text):
    t = text.lower().replace("#", "")
    t = EMOJI.sub("", t)
    t = re.sub(r"[^\w\sáéíóúüñ]", "", t)                 # punctuation
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                 # repeated characters
    return re.sub(r"\s+", " ", t).strip()


# =====================================================================
# Steps
# =====================================================================
raw = read_table(config.RAW_FILE, config.RAW_SHEET)
df = raw[[config.RAW_TEXT_COLUMN]].rename(columns={config.RAW_TEXT_COLUMN: "original"})
df.insert(0, config.ID_COLUMN, range(1, len(df) + 1))
record("Comments in the raw file", df)

df = df.dropna(subset=["original"])
df["original"] = df["original"].astype(str)
df = df[df["original"].str.strip() != ""]
record("After removing empty records", df)

df = df.drop_duplicates(subset=["original"])
record("After removing exact duplicates", df)

df = df[~df["original"].str.contains(r"@\w+", regex=True)]
record("After removing comments with user mentions", df)

df = df[~df["original"].str.contains(r"http\S+|www\S+", regex=True)]
record("After removing comments with URLs", df)

df[config.TEXT_COLUMN] = df["original"].apply(clean)
df = df[df[config.TEXT_COLUMN] != ""]
record("After removing comments left empty after cleaning", df)

# =====================================================================
# Save
# =====================================================================
out = df[[config.ID_COLUMN, config.TEXT_COLUMN]]
if str(config.INPUT_FILE).lower().endswith(".csv"):
    out.to_csv(config.INPUT_FILE, index=False)
else:
    out.to_excel(config.INPUT_FILE, index=False)

words = out[config.TEXT_COLUMN].str.split().str.len()
say("PREPROCESSING OF THE COMMENTS")
say("=" * 72)
say(f"Date: {datetime.now():%Y-%m-%d %H:%M}")
say(f"Raw file: {config.RAW_FILE}\n")
say(f"{'Step':<55}{'Left':>8}{'Removed':>10}")
previous = None
for name, n in steps:
    say(f"{name:<55}{n:>8}{'' if previous is None else previous - n:>10}")
    previous = n
first, last = steps[0][1], steps[-1][1]
say(f"\nRemoved in total: {first - last} ({100 * (first - last) / first:.1f}% of the raw comments)")
say("Steps are applied in order: a comment with a mention and a URL is counted only "
    "under mentions.\n")
say(f"Words per cleaned comment: mean = {words.mean():.1f}, median = {words.median():.0f}, "
    f"min = {words.min()}, max = {words.max()}, "
    f"IQR = {words.quantile(0.25):.0f}-{words.quantile(0.75):.0f}")
say(f"\nCleaned comments saved in: {config.INPUT_FILE}")
say.save(os.path.join(OUT, "summary.txt"))
