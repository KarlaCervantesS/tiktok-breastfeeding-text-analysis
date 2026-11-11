"""Classification with RoBERTuito models (pysentimiento), used by 02 and 03."""
import time

import numpy as np
import torch
from pysentimiento.preprocessing import preprocess_tweet
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def classify(texts, model_name):
    """Class probabilities for each text with a Hugging Face classification model.

    Texts are preprocessed as in RoBERTuito's training (pysentimiento): user
    handles, URLs, hashtags, emojis and repeated characters are normalized.
    The model runs in evaluation mode, so the results are deterministic.

    Returns (probabilities [n_texts x n_classes], class labels in lower case,
    number of texts longer than config.MAX_LENGTH tokens, which are truncated).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(config.DEVICE).eval()
    labels = [str(model.config.id2label[i]).lower() for i in range(model.config.num_labels)]

    texts = [preprocess_tweet(str(t), lang="es") for t in texts]
    probs, t0 = [], time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), config.BATCH_SIZE):
            enc = tokenizer(texts[start:start + config.BATCH_SIZE], padding=True,
                            truncation=True, max_length=config.MAX_LENGTH,
                            return_tensors="pt").to(config.DEVICE)
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
            print(f"  {min(start + config.BATCH_SIZE, len(texts))} of {len(texts)} "
                  f"({time.time() - t0:.0f} s)", end="\r")
    print()
    n_truncated = sum(len(ids) > config.MAX_LENGTH
                      for ids in tokenizer(texts, truncation=False)["input_ids"])
    return np.vstack(probs), labels, n_truncated
