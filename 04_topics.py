"""
4. Topics of the comments with BERTopic.

Steps
    1. Each comment is converted into an embedding (config.EMBEDDING_MODEL).
       Only light cleaning is applied (URLs, @mentions, laughter, config.ANONYMIZE);
       stop words are NOT removed, because the model needs whole sentences.
    2. UMAP reduces the embeddings to config.N_COMPONENTS dimensions.
    3. HDBSCAN groups them into topics; comments that do not fit any topic are
       outliers (topic -1). Minimum size config.MIN_CLUSTER_SIZE, maximum size
       config.MAX_CLUSTER_FRAC of the comments, min_samples config.MIN_SAMPLES.
    4. Each topic is described by its class-based TF-IDF keywords (stop words removed
       and words lemmatized with spaCy; this changes only the keywords, not the
       topics), by KeyBERTInspired keywords and by representative comments.
    5. If 02_sentiment.py was run before, the sentiment of each topic is reported.
    6. 2-D map of the comments coloured by topic (visualization only).

Topics are numbered from config.TOPIC_NUMBER_START (T1, T2, ...); outliers = -1.

Outputs (folder <OUTPUT_DIR>/topics/)
    topics.xlsx        sheets: topics (keywords, representative comments), comments
                       (each comment with its topic), topic_sentiment, parameters
    topics.tex         LaTeX table of the topics
    topic_map.pdf/.png 2-D map
    summary.txt
    embeddings_*.npy   cached embeddings (reused if the texts do not change)
"""
import hashlib
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

import config
from utils import Log, anonymize, load_comments, output_dir, package_versions, strip_accents

OUT = output_dir("topics")
SENTIMENT_FILE = os.path.join(config.OUTPUT_DIR, "sentiment", "sentiment_predictions.xlsx")
say = Log()
MAP_TOPICS = 8
MAP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#6250d6", "#e34948"]


# =====================================================================
# 1. Text processing
# =====================================================================
def clean_for_embedding(text):
    """Light cleaning: the sentence stays natural."""
    t = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", str(text))
    t = re.sub(r"\b(?:[jJ][aAeEiI]){2,}[jJ]?\b", "jaja", t)
    t = anonymize(t)
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)                      # "siiiiii" -> "siii"
    return re.sub(r"\s+", " ", t).strip()


_NLP, _STOP = None, None


def keyword_tokenizer(text):
    """Keyword tokens: stop words removed, lemmatized with spaCy, accents ignored."""
    global _NLP, _STOP
    if _NLP is None:
        import spacy
        from spacy.lang.es.stop_words import STOP_WORDS
        _NLP = spacy.load(config.SPACY_MODEL, disable=["parser", "ner"])
        _STOP = {strip_accents(w.lower()) for w in STOP_WORDS} | set(config.TOPIC_EXTRA_STOPWORDS)
    words = []
    for tok in _NLP(text):
        if tok.is_alpha and strip_accents(tok.lower_) not in _STOP:
            words.extend(tok.lemma_.split())
    words = [config.LEMMA_FIXES.get(w, w) for w in (strip_accents(w.lower()) for w in words)]
    return [w for w in words if len(w) > 2 and w not in _STOP]


# =====================================================================
# 2. Data and embeddings
# =====================================================================
df, n_file = load_comments()
df["doc"] = df[config.TEXT_COLUMN].apply(clean_for_embedding)
df = df[df["doc"] != ""].reset_index(drop=True)
n = len(df)
docs = df["doc"].tolist()
max_size = int(config.MAX_CLUSTER_FRAC * n)

say("TOPICS (BERTopic)")
say("=" * 72)
say(f"Input: {config.INPUT_FILE} ({n_file} rows; {n} with text)")
say(f"Embedding model: {config.EMBEDDING_MODEL}")
say(f"UMAP: n_neighbors = {config.N_NEIGHBORS}, n_components = {config.N_COMPONENTS}, "
    f"min_dist = 0, cosine, seed = {config.SEED}")
say(f"HDBSCAN: min_cluster_size = {config.MIN_CLUSTER_SIZE}, min_samples = "
    f"{config.MIN_SAMPLES}, max_cluster_size = {max_size}, euclidean, eom\n")

embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
key = hashlib.md5(("\n".join(docs) + config.EMBEDDING_MODEL).encode("utf-8")).hexdigest()[:12]
cache = os.path.join(OUT, f"embeddings_{key}.npy")
if os.path.exists(cache):
    emb = np.load(cache)
else:
    emb = embedder.encode(docs, batch_size=32, show_progress_bar=True)
    np.save(cache, emb)
emb = np.asarray(emb, dtype=np.float32)

# =====================================================================
# 3. BERTopic
# =====================================================================
model = BERTopic(
    embedding_model=embedder,
    umap_model=UMAP(n_neighbors=config.N_NEIGHBORS, n_components=config.N_COMPONENTS,
                    min_dist=0.0, metric="cosine", random_state=config.SEED),
    hdbscan_model=HDBSCAN(min_cluster_size=config.MIN_CLUSTER_SIZE,
                          min_samples=config.MIN_SAMPLES, metric="euclidean",
                          cluster_selection_method="eom", max_cluster_size=max_size,
                          prediction_data=True),
    vectorizer_model=CountVectorizer(tokenizer=keyword_tokenizer, token_pattern=None,
                                     lowercase=False, ngram_range=(1, 2), min_df=2),
    representation_model={"KeyBERT": KeyBERTInspired()},
    top_n_words=config.TOP_N_WORDS, calculate_probabilities=False)
raw_topics, _ = model.fit_transform(docs, embeddings=emb)


def renumber(t):
    return -1 if t == -1 else t + config.TOPIC_NUMBER_START


df["topic"] = [renumber(t) for t in raw_topics]
info = model.get_topic_info()
topic_table = pd.DataFrame({
    "topic": [renumber(t) for t in info["Topic"].astype(int)],
    "n": info["Count"].astype(int),
    "% comments": (100 * info["Count"] / n).round(1),
    "keywords": info["Representation"].apply(lambda w: ", ".join(x for x in w if x)),
    "keybert": info["KeyBERT"].apply(lambda w: ", ".join(x for x in w if x)),
    "representative_comments": info["Representative_Docs"].apply(" ||| ".join),
})
topic_table.insert(1, "label", ["Outliers" if t == -1 else f"T{t}" for t in topic_table["topic"]])
n_out = int((df["topic"] == -1).sum())
say(f"Topics: {int((topic_table['topic'] != -1).sum())}; outliers: {n_out} "
    f"({100 * n_out / n:.1f}%)")
say(topic_table[["label", "n", "% comments", "keywords"]].to_string(index=False))
say()

# =====================================================================
# 4. Sentiment of each topic (if 02_sentiment.py was run)
# =====================================================================
topic_sent = None
if os.path.exists(SENTIMENT_FILE):
    sent = pd.read_excel(SENTIMENT_FILE)[[config.ID_COLUMN, "sentiment"]]
    df = df.merge(sent, on=config.ID_COLUMN, how="left")
    classes = ["Negative", "Neutral", "Positive"]
    ct = pd.crosstab(df["topic"], df["sentiment"]).reindex(columns=classes, fill_value=0)
    topic_sent = (100 * ct.div(ct.sum(axis=1), axis=0)).round(1)
    topic_sent.insert(0, "n", ct.sum(axis=1).astype(int))
    topic_sent.index = ["Outliers" if t == -1 else f"T{t}" for t in topic_sent.index]
    topic_sent.index.name, topic_sent.columns.name = "Topic", None
    say("SENTIMENT OF EACH TOPIC (% of the comments of the topic)")
    say(topic_sent.to_string())
    say()
else:
    say("(Run 02_sentiment.py first to obtain the sentiment of each topic.)\n")

# =====================================================================
# 5. 2-D map (visualization only; topics were found in the UMAP space above)
# =====================================================================
xy = UMAP(n_neighbors=config.N_NEIGHBORS, n_components=2, min_dist=0.1, metric="cosine",
          random_state=config.SEED).fit_transform(emb)
df["map_x"], df["map_y"] = xy[:, 0], xy[:, 1]
largest = [t for t in df["topic"].value_counts().index if t != -1][:MAP_TOPICS]
fig, ax = plt.subplots(figsize=(8, 6.5))
other = ~df["topic"].isin(largest)
ax.scatter(df.loc[other, "map_x"], df.loc[other, "map_y"], s=3, c="#c8c8c8",
           linewidths=0, label="Other topics / outliers")
for t, col in zip(largest, MAP_COLORS):
    m = df["topic"] == t
    ax.scatter(df.loc[m, "map_x"], df.loc[m, "map_y"], s=5, c=col, linewidths=0, label=f"T{t}")
    ax.text(df.loc[m, "map_x"].median(), df.loc[m, "map_y"].median(), f"T{t}", fontsize=9,
            weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
ax.set_xticks([]); ax.set_yticks([])
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(fontsize=7, markerscale=2.5, frameon=False, loc="best")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(os.path.join(OUT, f"topic_map.{ext}"), dpi=300)
plt.close(fig)

# =====================================================================
# 6. Save
# =====================================================================
params = pd.DataFrame({
    "parameter": ["embedding_model", "seed", "n_neighbors", "n_components", "min_dist",
                  "min_cluster_size", "min_samples", "max_cluster_size", "top_n_words"],
    "value": [config.EMBEDDING_MODEL, config.SEED, config.N_NEIGHBORS, config.N_COMPONENTS,
              0.0, config.MIN_CLUSTER_SIZE, config.MIN_SAMPLES, max_size, config.TOP_N_WORDS]})
cols = [config.ID_COLUMN, config.TEXT_COLUMN, "topic", "map_x", "map_y"] + \
       (["sentiment"] if "sentiment" in df.columns else [])
with pd.ExcelWriter(os.path.join(OUT, "topics.xlsx")) as w:
    topic_table.to_excel(w, sheet_name="topics", index=False)
    df[cols].to_excel(w, sheet_name="comments", index=False)
    if topic_sent is not None:
        topic_sent.to_excel(w, sheet_name="topic_sentiment")
    params.to_excel(w, sheet_name="parameters", index=False)
with open(os.path.join(OUT, "topics.tex"), "w", encoding="utf-8") as f:
    t = topic_table[topic_table["topic"] != -1][["label", "n", "% comments", "keywords"]]
    f.write(t.to_latex(index=False, escape=True, float_format="%.1f"))

say("Software: " + package_versions(["bertopic", "sentence-transformers", "umap-learn",
                                     "hdbscan", "spacy"]))
say(f"Results saved in: {OUT}/")
say.save(os.path.join(OUT, "summary.txt"))
