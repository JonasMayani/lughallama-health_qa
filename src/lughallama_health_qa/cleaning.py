from __future__ import annotations

import csv
import html
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MOJIBAKE_MARKER_RE = re.compile(
    r"[\u00c2\u00c3]|\u00e2\u20ac|[\u00c5\u00c6\u00c9\u00ca][\u0080-\uffff]|\ufffd"
)

_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u00ab": '"',
        "\u00bb": '"',
    }
)

_COMMON_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u009c": '"',
    "\u00e2\u20ac\ufffd": '"',
    "\u00e2\u20ac\u00a6": "...",
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a2": "-",
    "\u00c2\u00a0": " ",
    "\u00c2": "",
    "\u00ef\u00bb\u00bf": "",
    "\u00ef\u00bf\u00bd": "",
    "\u00c9\u203a": "\u025b",
    "\u00c9\u201d": "\u0254",
    "\u00c9\u009b": "\u025b",
    "\u00c9\u0094": "\u0254",
    "\u00c6\u2020": "\u0186",
    "\u00c6\u0090": "\u0190",
    "\u00c5\u2039": "\u014b",
    "\u00c5\u0160": "\u014a",
    "\u00c9\u00b2": "\u0272",
    "\u00c9\u00a3": "\u0263",
    "\u00ca\u2039": "\u028b",
    "\u00c9\u2122": "\u0259",
    "\u00c9\u201c": "\u0253",
    "\u00c9\u2013": "\u0256",
    "\u00c9\u2014": "\u0257",
    "\u00c9\u00a6": "\u0266",
    "\u00c9\u00a8": "\u0268",
    "\u00c9\u00a9": "\u0269",
}

_PROMPT_SUFFIX_PATTERNS = [
    re.compile(
        r"\s*,?\s*please\s+answer(?:\s+this)?(?:\s+question)?"
        r"(?:\s+(?:using\s+)?simple\s+medical\s+terms|\s+in\s+detail|\s+clearly|\s+completely)?"
        r"\s*\.?\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\s*,?\s*answer\s+(?:in\s+detail|clearly|briefly)\s*\.?\s*$", flags=re.IGNORECASE),
]

_SUBSET_FIXES = {
    "aka": "Aka_Gha",
    "akan": "Aka_Gha",
    "aka_gha": "Aka_Gha",
    "twi": "Aka_Gha",
    "amh": "Amh_Eth",
    "amh_eth": "Amh_Eth",
    "amharic": "Amh_Eth",
    "eng": "Eng_Uga",
    "english": "Eng_Uga",
    "eng_eth": "Eng_Eth",
    "eng_ethiopia": "Eng_Eth",
    "eng_gha": "Eng_Gha",
    "eng_ghana": "Eng_Gha",
    "eng_ken": "Eng_Ken",
    "eng_kenya": "Eng_Ken",
    "eng_uga": "Eng_Uga",
    "eng_uganda": "Eng_Uga",
    "lug": "Lug_Uga",
    "lug_uga": "Lug_Uga",
    "luganda": "Lug_Uga",
    "swa": "Swa_Ken",
    "swa_ken": "Swa_Ken",
    "swahili": "Swa_Ken",
    "kiswahili": "Swa_Ken",
}


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _mojibake_score(text: str) -> int:
    return len(_MOJIBAKE_MARKER_RE.findall(text))


def _apply_common_mojibake_replacements(text: str) -> str:
    for bad, good in _COMMON_MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def _try_roundtrip_fix(text: str, encoding: str) -> str:
    try:
        candidate = text.encode(encoding).decode("utf-8")
    except UnicodeError:
        return text
    if _mojibake_score(candidate) < _mojibake_score(text):
        return candidate
    return text


def fix_mojibake(text: str) -> str:
    """Repair common UTF-8 text decoded as Windows-1252/Latin-1."""
    if not text:
        return ""

    try:
        from ftfy import fix_text

        fixed = fix_text(text)
    except Exception:
        fixed = text

    fixed = _apply_common_mojibake_replacements(fixed)
    for _ in range(2):
        before = fixed
        fixed = _try_roundtrip_fix(fixed, "cp1252")
        fixed = _try_roundtrip_fix(fixed, "latin1")
        fixed = _apply_common_mojibake_replacements(fixed)
        if fixed == before:
            break
    return fixed


def strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    quote_pairs = {('"', '"'), ("'", "'")}
    for _ in range(2):
        if len(text) >= 2 and (text[0], text[-1]) in quote_pairs:
            text = text[1:-1].strip()
        else:
            break
    text = re.sub(r'""+', '"', text)
    text = re.sub(r"''+", "'", text)
    return text


def normalize_spacing_and_punctuation(text: str) -> str:
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\.(?=[A-Z])", ". ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", text)
    return text.strip()


def clean_text(value: object, cfg: dict[str, Any] | None = None) -> str:
    cleaning_cfg = (cfg or {}).get("cleaning", {})
    text = "" if _is_missing(value) else str(value)
    text = html.unescape(text)
    if cleaning_cfg.get("fix_mojibake", True):
        text = fix_mojibake(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub(" ", text)
    if cleaning_cfg.get("strip_urls", True):
        text = _URL_RE.sub(" ", text)
    if cleaning_cfg.get("normalize_quotes", True):
        text = text.translate(_QUOTE_TRANSLATION)
        text = strip_wrapping_quotes(text)
    text = normalize_spacing_and_punctuation(text)
    return text


def clean_question(value: object, cfg: dict[str, Any] | None = None) -> str:
    text = clean_text(value, cfg)
    if (cfg or {}).get("cleaning", {}).get("remove_prompt_suffixes", True):
        for pattern in _PROMPT_SUFFIX_PATTERNS:
            text = pattern.sub("", text).strip()
    return normalize_spacing_and_punctuation(text)


def clean_answer(value: object, cfg: dict[str, Any] | None = None) -> str:
    return clean_text(value, cfg)


def normalize_subset(value: object, valid_subsets: set[str]) -> str:
    subset = clean_text(value, {"cleaning": {"strip_urls": False}}).strip()
    if subset in valid_subsets:
        return subset
    return _SUBSET_FIXES.get(subset.lower(), subset)


def read_qa_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    read_kwargs = {
        "dtype": str,
        "keep_default_na": False,
        "na_filter": False,
    }
    try:
        return pd.read_csv(path, **read_kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(
            path,
            **read_kwargs,
            engine="python",
            doublequote=True,
            escapechar="\\",
            on_bad_lines="warn",
        )


def clean_qa_dataframe(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    split: str,
    require_output: bool,
    preserve_rows: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cols = cfg["dataset"]
    id_col = cols.get("id_col", "ID")
    input_col = cols.get("input_col", "input")
    output_col = cols.get("output_col", "output")
    subset_col = cols.get("subset_col", "subset")
    valid_subsets = set(cols.get("languages", []))
    cleaning_cfg = cfg.get("cleaning", {})

    work = df.copy()
    issues: list[dict[str, Any]] = []

    def row_id(idx: Any) -> str:
        if id_col in work.columns:
            return str(work.at[idx, id_col])
        return f"{split}_{idx}"

    def record(idx: Any, field: str, before: object, after: object, issue: str) -> None:
        issues.append(
            {
                "split": split,
                "row_id": row_id(idx),
                "field": field,
                "issue": issue,
                "before": str(before)[:250],
                "after": str(after)[:250],
            }
        )

    for column, cleaner in [(input_col, clean_question), (output_col, clean_answer)]:
        if column not in work.columns:
            continue
        original = work[column].copy()
        work[column] = work[column].map(lambda value: cleaner(value, cfg))
        changed = work[column] != original
        for idx in work.index[changed]:
            record(idx, column, original.at[idx], work.at[idx, column], "text_normalized")

    if subset_col in work.columns:
        original_subset = work[subset_col].copy()
        work[subset_col] = work[subset_col].map(lambda value: normalize_subset(value, valid_subsets))
        changed = work[subset_col] != original_subset
        for idx in work.index[changed]:
            record(idx, subset_col, original_subset.at[idx], work.at[idx, subset_col], "subset_normalized")

    drop_mask = pd.Series(False, index=work.index)
    min_input_chars = int(cleaning_cfg.get("min_input_chars", 3))
    min_output_chars = int(cleaning_cfg.get("min_output_chars", 3))

    if input_col in work.columns:
        bad_input = work[input_col].str.len() < min_input_chars
        drop_mask |= bad_input
        for idx in work.index[bad_input]:
            record(idx, input_col, work.at[idx, input_col], "", "empty_or_too_short_input")

    if require_output and output_col in work.columns:
        bad_output = work[output_col].str.len() < min_output_chars
        drop_mask |= bad_output
        for idx in work.index[bad_output]:
            record(idx, output_col, work.at[idx, output_col], "", "empty_or_too_short_output")

    if subset_col in work.columns and valid_subsets:
        invalid_subset = ~work[subset_col].isin(valid_subsets)
        drop_mask |= invalid_subset
        for idx in work.index[invalid_subset]:
            record(idx, subset_col, work.at[idx, subset_col], "", "invalid_subset")

    if preserve_rows:
        drop_mask = pd.Series(False, index=work.index)

    if drop_mask.any():
        work = work.loc[~drop_mask].copy()

    if (
        split == "train"
        and require_output
        and cleaning_cfg.get("drop_train_duplicates", True)
        and {input_col, output_col, subset_col}.issubset(work.columns)
    ):
        before = len(work)
        duplicate_mask = work.duplicated(subset=[input_col, output_col, subset_col], keep="first")
        for idx in work.index[duplicate_mask]:
            record(idx, input_col, work.at[idx, input_col], "", "duplicate_train_row")
        work = work.loc[~duplicate_mask].copy()
        if before != len(work):
            issues.append(
                {
                    "split": split,
                    "row_id": "*",
                    "field": "*",
                    "issue": "duplicates_removed",
                    "before": str(before),
                    "after": str(len(work)),
                }
            )

    return work.reset_index(drop=True), issues


def write_clean_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL, lineterminator="\n")
