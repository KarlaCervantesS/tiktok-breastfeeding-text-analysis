"""
Configuration shared by all the scripts. Edit this file to use your own data.

Input: one table (.xlsx or .csv) with one row per comment and at least
    - an identifier column (ID_COLUMN); if it does not exist, row numbers are used
    - a text column with the preprocessed comments (TEXT_COLUMN)

The values below are the ones used in our study (Spanish TikTok comments about
exclusive breastfeeding). Lists marked "STUDY-SPECIFIC" should be adapted to
your own topic and language.
"""

# =====================================================================
# 0. Preprocessing (00_preprocess.py)
# =====================================================================
# Raw comments as downloaded (e.g. with a TikTok comment scraper): one row per
# comment. 00_preprocess.py writes the cleaned comments to INPUT_FILE below.
RAW_FILE = "raw_comments.xlsx"   # .xlsx or .csv
RAW_SHEET = 0
RAW_TEXT_COLUMN = "text"         # column with the original comment

# =====================================================================
# Input and output of the analyses
# =====================================================================
INPUT_FILE = "comments.xlsx"     # .xlsx or .csv (output of 00_preprocess.py)
INPUT_SHEET = 0                  # sheet name or index (only for .xlsx)
ID_COLUMN = "id"
TEXT_COLUMN = "text"             # preprocessed comments (in Spanish)
OUTPUT_DIR = "results"           # each script writes to a subfolder of this folder

SEED = 2026                      # random seed (UMAP, word-cloud layout)
DEVICE = "cpu"                   # "cpu" (reproducible), "cuda" or "mps"

# Anonymization: regular expressions replaced by a generic term before the
# vocabulary and topic analyses, e.g. {r"\bjohn\s?doe\b": "influencer"}.
# Leave empty if your texts are already anonymized.
ANONYMIZE = {}

# =====================================================================
# 1. Vocabulary (01_vocabulary.py)
# =====================================================================
TOP_N = 10                       # rows of the term / bigram / trigram tables
WORDCLOUD_WORDS = 100            # terms shown in the word cloud
KEEP_NUMBERS_IN_NGRAMS = False   # True: "6 meses" is kept in pairs/triplets

# STUDY-SPECIFIC: words that refer to the subject of the corpus and appear in
# many comments (removed by stem from single terms)
DOMAIN_STOPWORDS = ["bebe", "bebes", "bb", "lactancia", "amamantar", "lactar"]
DOMAIN_STOP_IN_NGRAMS = False    # False: kept in pairs/triplets ("lactancia materna")

# Filler words of social-media comments (added to the Snowball stop-word list)
FILLER_WORDS = ["q", "k", "xq", "pq", "porq", "x", "d", "tb", "tmb", "pa", "ps", "pues",
                "osea", "asi", "ahi", "aqui", "ya", "jaja", "jeje", "ok", "va", "hola",
                "vdd", "bueno", "verdad", "solo", "bien", "super", "igual", "mismo",
                "cada", "casi", "cosas", "caso", "veces", "ahora", "despues", "tan",
                "aun", "entonces", "vez", "parte", "luego", "ay"]

# Words of the Snowball list that are KEPT because they change meaning
# (negation, contrast, quantity); they are still excluded from the word cloud
WORDS_TO_KEEP = ["no", "ni", "nada", "sin", "contra", "pero", "porque", "mas", "muy",
                 "mucho", "muchos", "poco", "otro", "otra", "otros", "otras", "durante",
                 "hasta", "sobre", "antes", "tanto", "o"]

# STUDY-SPECIFIC: words removed only from the word cloud and the top-terms table
# (very frequent terms and generic verbs); matched by exact word and by stem
CLOUD_ONLY_STOPWORDS = [
    "leche", "meses", "hijos", "años",
    "dar", "darle", "da", "doy", "di", "dio", "hacer", "tener", "decir", "dice", "dijo",
    "ir", "ver", "poder", "puede", "pude", "pueden", "querer", "quiere", "quise",
    "quiero", "deber", "debe", "deberia", "saber", "ser", "pasa", "creo", "queda",
    "hago", "hice", "digo", "habla", "medio", "daba", "dan", "punto", "mundo",
    "demasiado", "dos", "todas", "todos", "muchas", "siempre", "nunca", "menos"]

# Stems that the Snowball stemmer splits but belong to the same word
STEM_MERGE = {"lechit": "lech", "anos": "ano", "dias": "dia", "muj": "mujer",
              "prim": "primer", "amam": "amamant", "doli": "duel", "sali": "sal",
              "produccion": "produc"}

# Optional English labels for a translated word cloud: a .csv/.xlsx file with
# columns "stem" and "english" (see examples/wordcloud_translations_example.csv).
# None = only the Spanish word cloud is drawn.
TRANSLATION_FILE = None

# =====================================================================
# 2. Sentiment (02_sentiment.py) and 3. Emotions (03_emotions.py)
# =====================================================================
SENTIMENT_MODEL = "pysentimiento/robertuito-sentiment-analysis"
EMOTION_MODEL = "pysentimiento/robertuito-emotion-analysis"
MAX_LENGTH = 128                 # maximum input length of RoBERTuito (tokens)
BATCH_SIZE = 32

# Optional validation against human labels; None = no validation.
# The file must contain the human label (Negative / Neutral / Positive) and either
# the comment id (VALIDATION_ID_COLUMN) or its text (VALIDATION_TEXT_COLUMN).
VALIDATION_FILE = None
VALIDATION_SHEET = 0
VALIDATION_LABEL_COLUMN = "label"
VALIDATION_ID_COLUMN = "id"      # None if the validation file has no ids
# Used only when there are no ids: the text the annotator read and the column of
# the input table that contains that same text (e.g. a translation)
VALIDATION_TEXT_COLUMN = None
CORPUS_MATCH_COLUMN = None
FUZZY_MIN_SCORE = 0.85           # similarity (0-1) to accept a similar-text match
FUZZY_MARGIN = 0.05              # best match must beat the second best by this much

# =====================================================================
# 4. Topics (04_topics.py)
# =====================================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SPACY_MODEL = "es_core_news_sm"  # lemmatization of the topic keywords
N_NEIGHBORS = 15                 # UMAP
N_COMPONENTS = 5                 # UMAP
MIN_CLUSTER_SIZE = 40            # HDBSCAN: minimum comments per topic
MIN_SAMPLES = 5                  # HDBSCAN: lower values -> fewer outliers
MAX_CLUSTER_FRAC = 0.25          # HDBSCAN: maximum topic size (fraction of comments)
TOP_N_WORDS = 10                 # keywords per topic
TOPIC_NUMBER_START = 1           # topics are labeled T1, T2, ... (0 for T0, T1, ...)

# Extra stop words for the topic keywords only (fillers, generic verbs)
TOPIC_EXTRA_STOPWORDS = ["jaja", "jeje", "q", "k", "xq", "pq", "porq", "x", "d", "tb",
                         "tmb", "pa", "ps", "pues", "osea", "asi", "ahi", "aqui", "si",
                         "no", "ya", "mas", "solo", "bien", "cosa", "hacer", "decir",
                         "ir", "ver", "dar", "tener", "ser", "estar", "haber", "poder",
                         "querer", "saber", "vez", "ok", "va", "hola", "el", "ella",
                         "jajar", "bb"]
# Corrections of frequent errors of spaCy's small Spanish lemmatizer
LEMMA_FIXES = {"maternar": "materno", "materna": "materno", "bendigar": "bendecir",
               "hermosa": "hermoso", "hermosar": "hermoso", "hagar": "hacer",
               "tranquilar": "tranquilo", "gemela": "gemelo", "amantar": "amamantar"}

# =====================================================================
# 5. Sentiment by topic group (05_sentiment_by_group.py)
# =====================================================================
# .csv/.xlsx with columns "topic" (number as in 04_topics.py), "label" and "group";
# see examples/topic_groups_example.csv
TOPIC_GROUPS_FILE = "topic_groups.csv"
