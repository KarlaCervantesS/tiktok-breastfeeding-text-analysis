"""Helper functions shared by the scripts."""
import os
import re
import unicodedata
from importlib.metadata import version

import pandas as pd

import config


class Log:
    """Print lines and keep them to save a summary file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(str(text))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def output_dir(name):
    """Create and return config.OUTPUT_DIR/name."""
    path = os.path.join(config.OUTPUT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def read_table(path, sheet=0):
    """Read a .csv or .xlsx file."""
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


def load_comments():
    """Input table with an id column and non-empty texts (as strings)."""
    df = read_table(config.INPUT_FILE, config.INPUT_SHEET)
    if config.TEXT_COLUMN not in df.columns:
        raise KeyError(f"Column '{config.TEXT_COLUMN}' not found in {config.INPUT_FILE}. "
                       f"Available columns: {list(df.columns)}")
    if config.ID_COLUMN not in df.columns:
        df.insert(0, config.ID_COLUMN, range(1, len(df) + 1))
    n_file = len(df)
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].fillna("").astype(str)
    df = df[df[config.TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)
    return df, n_file


def anonymize(text):
    """Replace the patterns of config.ANONYMIZE by their generic terms."""
    for pattern, repl in config.ANONYMIZE.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def package_versions(names):
    out = []
    for name in names:
        try:
            out.append(f"{name} {version(name)}")
        except Exception:
            out.append(f"{name} (not found)")
    return ", ".join(out)
