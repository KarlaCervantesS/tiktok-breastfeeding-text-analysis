"""
1. Vocabulary of the comments: word cloud, most frequent terms, pairs and triplets.

Text processing
    - lower case; URLs, @mentions and punctuation removed; laughter normalized
      ("jajaja" -> "jaja"); patterns in config.ANONYMIZE replaced;
    - accents ignored ("bebe" and "bebé" are the same word);
    - pairs/triplets never cross punctuation (a comma or period ends a phrase);
    - stop words: Snowball Spanish list (NLTK) + config.FILLER_WORDS, minus
      config.WORDS_TO_KEEP; config.DOMAIN_STOPWORDS are removed by stem;
    - stemming: Snowball Spanish stemmer, with the merges in config.STEM_MERGE;
      each stem is displayed with its most frequent original word;
    - numbers: always removed from single terms; kept in pairs/triplets only if
      config.KEEP_NUMBERS_IN_NGRAMS is True.

Frequencies: n = total occurrences; comments = number (and %) of comments that
contain the term at least once.

Outputs (folder <OUTPUT_DIR>/vocabulary/)
    wordcloud_spanish.pdf/.png, wordcloud_english.pdf/.png (if TRANSLATION_FILE),
    word_frequencies.xlsx, ngram_tables.tex, stop_words_used.txt, summary.txt
"""
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud

import config
from utils import Log, anonymize, load_comments, output_dir, read_table, strip_accents

OUT = output_dir("vocabulary")
say = Log()


def clean_word(w):
    """Lower case, no accents and ñ -> n (same normalization for texts and lists)."""
    return strip_accents(str(w).lower()).replace("ñ", "n")


# =====================================================================
# 1. Stop words and stemming
# =====================================================================
try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords", quiet=True)
snowball_list = stopwords.words("spanish")
stemmer = SnowballStemmer("spanish")


def stem_word(w):
    s = stemmer.stem(w)
    return config.STEM_MERGE.get(s, s)


stop_word_list = {clean_word(w) for w in snowball_list + config.FILLER_WORDS}
stop_word_list -= {clean_word(w) for w in config.WORDS_TO_KEEP}
domain_stems = {stem_word(clean_word(w)) for w in config.DOMAIN_STOPWORDS}
cloud_only_words = {clean_word(w) for w in config.WORDS_TO_KEEP + config.CLOUD_ONLY_STOPWORDS}
cloud_only_stems = {stem_word(clean_word(w)) for w in config.CLOUD_ONLY_STOPWORDS}

say("STOP WORDS")
say(f"  Snowball Spanish list: {len(snowball_list)} | fillers: {len(config.FILLER_WORDS)} | "
    f"kept: {len(config.WORDS_TO_KEEP)} | final list: {len(stop_word_list)} words")
say(f"  Domain stop words (removed by stem): {', '.join(config.DOMAIN_STOPWORDS)}")
say()


# =====================================================================
# 2. Tokenization
# =====================================================================
def tokenize(text):
    """Tokens (word, stem, original spelling) in their order; "|" = phrase boundary."""
    t = anonymize(str(text).lower())
    t = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", t)
    t = re.sub(r"\b(?:j[aeiou]){2,}j?\b", "jaja", t)
    t = re.sub(r"[.,;:!?¡¿()\"\n]+", " | ", t)
    pattern = r"\||\d+|[^\W\d_]+" if config.KEEP_NUMBERS_IN_NGRAMS else r"\||[^\W\d_]+"
    tokens = []
    for w in re.findall(pattern, t):
        if w == "|":
            tokens.append(("|", "|", "|"))
        elif w.isdigit():
            tokens.append((w, w, w))
        else:
            cw = clean_word(w)
            tokens.append((cw, stem_word(cw), w))
    return tokens


def is_stop(token, domain=True):
    word, stem, _ = token
    if word.isdigit():
        return False
    return (word == "|" or word in stop_word_list or len(word) < 2
            or (domain and stem in domain_stems))


# =====================================================================
# 3. Counting
# =====================================================================
def count_terms(docs, n):
    """Count terms (n = 1), pairs (n = 2) or triplets (n = 3) of stems."""
    total, in_docs, spellings = Counter(), Counter(), defaultdict(Counter)
    for tokens in docs:
        seen = set()
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i + n]
            stems = tuple(s for _, s, _ in gram)
            if n == 1 and (gram[0][0].isdigit() or is_stop(gram[0])
                           or gram[0][0] in cloud_only_words or gram[0][1] in cloud_only_stems):
                continue
            if n > 1 and (is_stop(gram[0], config.DOMAIN_STOP_IN_NGRAMS)
                          or is_stop(gram[-1], config.DOMAIN_STOP_IN_NGRAMS)
                          or "|" in stems or gram[-1][0].isdigit()
                          or len(set(stems)) == 1):
                continue
            total[stems] += 1
            seen.add(stems)
            spellings[stems][" ".join(s for _, _, s in gram)] += 1
        in_docs.update(seen)
    rows = [{"term": spellings[k].most_common(1)[0][0], "stem": " ".join(k), "n": c,
             "comments": in_docs[k], "% comments": round(100 * in_docs[k] / len(docs), 1)}
            for k, c in total.most_common()]
    return pd.DataFrame(rows, columns=["term", "stem", "n", "comments", "% comments"])


# =====================================================================
# 4. Word cloud
# =====================================================================
CLOUD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#d99400", "#e87ba4",
                "#008300", "#6250d6", "#e34948"]


def save_wordcloud(freq, name):
    """Elliptical word cloud; size = frequency, colour is decorative."""
    width, height = 1800, 1100
    y, x = np.ogrid[:height, :width]
    outside = ((x - width / 2) / (width / 2 - 20)) ** 2 + \
              ((y - height / 2) / (height / 2 - 20)) ** 2 > 1
    mask = np.where(outside, 255, 0).astype(np.uint8)

    def color_func(word, *args, **kwargs):
        return CLOUD_COLORS[sum(map(ord, word)) % len(CLOUD_COLORS)]

    wc = WordCloud(width=width, height=height, mask=mask, background_color="white",
                   max_words=config.WORDCLOUD_WORDS, prefer_horizontal=0.9,
                   random_state=config.SEED, collocations=False, relative_scaling=0.5,
                   margin=4, color_func=color_func).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight",
                    pad_inches=0.05)
    plt.close(fig)


# =====================================================================
# 5. Run
# =====================================================================
df, _ = load_comments()
docs = [tokenize(t) for t in df[config.TEXT_COLUMN]]
docs = [d for d in docs if any(w != "|" for w, _, _ in d)]
n_words = sum(w != "|" and not w.isdigit() for d in docs for w, _, _ in d)

uni, bi, tri = (count_terms(docs, k) for k in (1, 2, 3))
say(f"Comments: {len(docs)} | words: {n_words} | distinct terms (no stop words): {len(uni)}")
say(f"\nTop-{config.TOP_N} terms:\n{uni.head(config.TOP_N).to_string(index=False)}")
say(f"\nTop-{config.TOP_N} pairs:\n{bi.head(config.TOP_N).to_string(index=False)}")
say(f"\nTop-{config.TOP_N} triplets:\n{tri.head(config.TOP_N).to_string(index=False)}")

top = uni.head(config.WORDCLOUD_WORDS).copy()
save_wordcloud(dict(zip(top["term"], top["n"])), "wordcloud_spanish")

if config.TRANSLATION_FILE:
    tr = read_table(config.TRANSLATION_FILE).dropna(subset=["stem", "english"])
    top["english"] = top["stem"].map(dict(zip(tr["stem"].astype(str), tr["english"].astype(str))))
    missing = top[top["english"].isna()]
    if len(missing):
        say(f"\nWARNING: {len(missing)} terms without translation (shown in Spanish):")
        say(missing[["term", "stem", "n"]].to_string(index=False))
    top["english"] = top["english"].fillna(top["term"])
    save_wordcloud(top.groupby("english")["n"].sum().to_dict(), "wordcloud_english")

# =====================================================================
# 6. Save
# =====================================================================
with pd.ExcelWriter(os.path.join(OUT, "word_frequencies.xlsx")) as w:
    top.to_excel(w, sheet_name="wordcloud_terms", index=False)
    bi.head(50).to_excel(w, sheet_name="pairs", index=False)
    tri.head(50).to_excel(w, sheet_name="triplets", index=False)

with open(os.path.join(OUT, "ngram_tables.tex"), "w", encoding="utf-8") as f:
    for table, kind, title in [(uni, "terms", "terms"), (bi, "bigrams", "pairs of terms"),
                               (tri, "trigrams", "triplets of terms")]:
        t = table.head(config.TOP_N)[["term", "n", "% comments"]].copy()
        t.insert(0, "Rank", range(1, len(t) + 1))
        t.columns = ["Rank", "Terms", "n", "\\% comments"]
        f.write("\\begin{table}[!htbp]\n\\centering\n"
                f"\\caption{{Top-{config.TOP_N} most frequent {title}.}}\n"
                f"\\label{{tab:top-{kind}}}\n"
                + t.to_latex(index=False, escape=False, float_format="%.1f")
                + "\\end{table}\n\n")

with open(os.path.join(OUT, "stop_words_used.txt"), "w", encoding="utf-8") as f:
    f.write("Stop words (accents removed):\n" + ", ".join(sorted(stop_word_list)) + "\n\n")
    f.write("Domain stop words (by stem): " + ", ".join(config.DOMAIN_STOPWORDS) + "\n")
    f.write("Kept words: " + ", ".join(config.WORDS_TO_KEEP) + "\n")
    f.write("Stems merged: " + ", ".join(f"{a} -> {b}" for a, b in config.STEM_MERGE.items()) + "\n")
    f.write("Removed only from the word cloud: " + ", ".join(config.CLOUD_ONLY_STOPWORDS) + "\n")

say(f"\nResults saved in: {OUT}/")
say.save(os.path.join(OUT, "summary.txt"))
