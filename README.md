# Text Analysis of Spanish-Language TikTok Comments on Breastfeeding Using Large Language Models

Code accompanying the article:

> Seda C, Cervantes-Sanmiguel KI, Chavez-Santoscoy RA, Vazquez AR. Text Analysis of
> Spanish-Language TikTok Comments on Breastfeeding Using Large Language Models.


**Authors:** Clarisa Seda, Karla Isabel Cervantes-Sanmiguel
([ORCID 0009-0008-1301-466X](https://orcid.org/0009-0008-1301-466X)),
Rocio Alejandra Chavez-Santoscoy ([ORCID 0000-0002-4551-1862](https://orcid.org/0000-0002-4551-1862)), and Alan R. Vazquez
([ORCID 0000-0002-3658-0911](https://orcid.org/0000-0002-3658-0911); corresponding author).
School of Engineering and Sciences, Tecnologico de Monterrey, Mexico.

The scripts analyze a collection of Spanish-language comments (in our study, TikTok
comments about exclusive breastfeeding). **The data used in the study are not
shared**; the code can be applied to any table of Spanish comments.

## Pipeline

| Script | Analysis | Main methods |
|---|---|---|
| `00_preprocess.py` | Removal of empty, duplicate, mention and URL comments; text cleaning; number of comments removed in each step | pandas, regular expressions |
| `01_vocabulary.py` | Word cloud and most frequent terms, pairs and triplets | Snowball stop words and stemmer (NLTK) |
| `02_sentiment.py` | Negative / neutral / positive sentiment, optional validation against human labels | RoBERTuito (`pysentimiento/robertuito-sentiment-analysis`) |
| `03_emotions.py` | Ekman's six basic emotions or no emotion | RoBERTuito (`pysentimiento/robertuito-emotion-analysis`) |
| `04_topics.py` | Topics, keywords, representative comments, sentiment per topic, 2-D map | BERTopic (sentence embeddings, UMAP, HDBSCAN, c-TF-IDF, KeyBERTInspired) |
| `05_sentiment_by_group.py` | Sentiment per group of topics (e.g. themes) and per topic | Stacked bar chart |

Run the scripts in numerical order. `00` produces the input of the analyses;
`02` must run before `03`, `04` and `05` if you want the sentiment of each emotion,
topic or group; `05` needs the outputs of `02` and `04`.

## Input

**Raw comments** (`RAW_FILE`, input of `00_preprocess.py`): one `.xlsx` or `.csv`
file with one row per comment and a column with the original text (`text` by
default, as in the files produced by common TikTok comment scrapers).

`00_preprocess.py` writes the **cleaned comments** (`INPUT_FILE`), the input of all
the analyses, with two columns:

| Column | Content |
|---|---|
| `id` | identifier of the comment (row number in the raw file) |
| `text` | cleaned comment in Spanish |

In our study, the cleaned comments were then reviewed manually to exclude comments
unrelated to the topic or containing personal names; this review is not part of the
code. Names of people can also be replaced by a generic term with `ANONYMIZE` in
`config.py`. If your comments are already cleaned, skip `00` and point `INPUT_FILE`
to your file. Column names, file names and all parameters are set in
**`config.py`**.

Optional files:
- **Human labels** for validating the sentiment model (`VALIDATION_FILE`): a label
  column (Negative / Neutral / Positive) and either the comment `id` or its text
  (matched to a column of the input).
- **Topic groups** (`TOPIC_GROUPS_FILE`) for `05`: columns `topic`, `label`,
  `group`; see `examples/topic_groups_example.csv` (the groups used in our study).
- **Word-cloud translations** (`TRANSLATION_FILE`): columns `stem`, `english`; see
  `examples/wordcloud_translations_example.csv`.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

The language models are downloaded from the Hugging Face Hub the first time they are
used. All analyses run on CPU; set `DEVICE` in `config.py` to `"cuda"` or `"mps"` to
use a GPU.

## Usage

```bash
python 00_preprocess.py
python 01_vocabulary.py
python 02_sentiment.py
python 03_emotions.py
python 04_topics.py
python 05_sentiment_by_group.py
```

Results are written to `results/` (one subfolder per analysis), including Excel
tables, LaTeX tables, figures (PDF and PNG) and a `summary.txt` file with everything
printed on screen.

## Adapting the code

- **Other topics**: edit the study-specific lists in `config.py`
  (`DOMAIN_STOPWORDS`, `CLOUD_ONLY_STOPWORDS`, `STEM_MERGE`, `LEMMA_FIXES`).
- **Topic granularity**: `MIN_CLUSTER_SIZE` controls the minimum size (and hence the
  number) of topics; `MIN_SAMPLES` controls the proportion of outliers;
  `MAX_CLUSTER_FRAC` limits the size of the largest topic.
- **Other languages**: the sentiment and emotion models, the stop-word list, the
  stemmer and the spaCy model are specific to Spanish and must be replaced.

## Notes

- Topic labels and the grouping of topics into themes reported in the article were
  generated with a large language model and reviewed by the authors; they are not
  produced by these scripts.
- Results of UMAP may vary slightly across operating systems and package versions
  even with a fixed seed.

## License

MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the article above and the models and methods it
relies on (RoBERTuito / pysentimiento, BERTopic, Sentence-Transformers, UMAP and
HDBSCAN).
