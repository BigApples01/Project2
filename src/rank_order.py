from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
MIN_N = 5
TOP_N_PLOT = 20

LIKERT_MAP = {
    "strongly disagree": 1,
    "disagree": 2,
    "somewhat disagree": 2,
    "neutral": 3,
    "neither agree nor disagree": 3,
    "somewhat agree": 4,
    "agree": 4,
    "strongly agree": 5,
    "very dissatisfied": 1,
    "dissatisfied": 2,
    "neither satisfied nor dissatisfied": 3,
    "satisfied": 4,
    "very satisfied": 5,
    "poor": 1,
    "fair": 2,
    "good": 3,
    "very good": 4,
    "excellent": 5,
    "least preferred": 1,
    "most preferred": 5,
}

IGNORE_COL_PATTERNS = [
    r"timestamp",
    r"time stamp",
    r"email",
    r"name",
    r"student id",
    r"id$",
    r"uid",
    r"comment",
    r"feedback",
    r"suggestion",
    r"open\s*ended",
    r"free\s*text",
    r"reason",
    r"why",
    r"department",
    r"program enrolled",
]

RATING_HINT_PATTERNS = [
    r"rate",
    r"rating",
    r"satisf",
    r"quality",
    r"effective",
    r"agree",
    r"prefer",
    r"overall",
    r"course",
    r"program",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def detect_dataset_file(data_dir: Path) -> Path:
    excel_files = sorted(data_dir.glob("*.xlsx"))
    csv_files = sorted(data_dir.glob("*.csv"))
    if excel_files:
        return excel_files[0]
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(f"No .xlsx or .csv file found in {data_dir}")


def choose_best_sheet(path: Path) -> str:
    workbook = pd.ExcelFile(path)
    best_sheet = workbook.sheet_names[0]
    best_score = -1.0

    for sheet in workbook.sheet_names:
        sample = pd.read_excel(path, sheet_name=sheet, nrows=200)
        rows, cols = sample.shape
        sheet_score = rows * 0.1 + cols
        sheet_name = sheet.lower()
        if any(k in sheet_name for k in ["response", "survey", "data", "results"]):
            sheet_score += 20
        if any(k in sheet_name for k in ["summary", "pivot", "chart"]):
            sheet_score -= 10
        if sheet_score > best_score:
            best_score = sheet_score
            best_sheet = sheet

    return best_sheet


def should_ignore_column(col_name: str) -> bool:
    name = col_name.strip().lower()
    return any(re.search(pattern, name) for pattern in IGNORE_COL_PATTERNS)


def contains_rating_hint(col_name: str) -> bool:
    name = col_name.strip().lower()
    return any(re.search(pattern, name) for pattern in RATING_HINT_PATTERNS)


def map_likert_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(normalize_text)

    numeric = pd.to_numeric(cleaned, errors="coerce")

    mapped = cleaned.map(LIKERT_MAP)

    pattern_values = []
    for value in cleaned:
        if not value:
            pattern_values.append(None)
            continue
        match = re.search(r"\b([1-9]|10)\b", value)
        pattern_values.append(float(match.group(1)) if match else None)

    pattern_numeric = pd.Series(pattern_values, index=series.index, dtype="float")

    combined = numeric.fillna(mapped).fillna(pattern_numeric)
    return pd.to_numeric(combined, errors="coerce")


def is_rating_column(col_name: str, col_data: pd.Series) -> bool:
    if should_ignore_column(col_name):
        return False

    non_null = col_data.dropna()
    if non_null.empty:
        return False

    numeric_values = map_likert_to_numeric(col_data).dropna()
    if numeric_values.empty:
        return False

    coverage = len(numeric_values) / len(non_null)
    if coverage < 0.6:
        return False

    unique_count = numeric_values.nunique()
    min_value = numeric_values.min()
    max_value = numeric_values.max()

    plausible_scale = (1 <= min_value <= 7) and (1 <= max_value <= 10)
    bounded_likertish = unique_count <= 10 and plausible_scale

    if bounded_likertish:
        return True

    return contains_rating_hint(col_name) and coverage >= 0.8


def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    ranking_rows: list[dict[str, object]] = []

    for column in df.columns:
        series = df[column]
        if not is_rating_column(column, series):
            continue

        numeric = map_likert_to_numeric(series)
        valid = numeric.dropna()

        if len(valid) < MIN_N:
            continue

        ranking_rows.append(
            {
                "item": str(column).strip(),
                "mean_score": float(valid.mean()),
                "n": int(len(valid)),
            }
        )

    if not ranking_rows:
        return pd.DataFrame(columns=["item", "mean_score", "n", "rank"])

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values(
        by=["mean_score", "n", "item"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1
    ranking_df["mean_score"] = ranking_df["mean_score"].round(3)

    return ranking_df[["item", "mean_score", "n", "rank"]]


def plot_rankings(ranking_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 8))

    if ranking_df.empty:
        plt.text(0.5, 0.5, "No rating/preference columns met threshold", ha="center", va="center")
        plt.axis("off")
    else:
        plot_df = ranking_df.head(TOP_N_PLOT).iloc[::-1]
        plt.barh(plot_df["item"], plot_df["mean_score"], color="#2b8cbe")
        plt.xlabel("Mean score")
        plt.ylabel("Program/Course/Question")
        plt.title("Rank order of programs/courses by student ratings/preferences")
        plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = detect_dataset_file(DATA_DIR)

    if dataset_path.suffix.lower() == ".xlsx":
        sheet_name = choose_best_sheet(dataset_path)
        df = pd.read_excel(dataset_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(dataset_path)

    rankings = compute_rankings(df)
    rankings.to_csv(OUTPUTS_DIR / "rank_order.csv", index=False)
    plot_rankings(rankings, OUTPUTS_DIR / "rank_order.png")

    print(f"Dataset: {dataset_path}")
    if dataset_path.suffix.lower() == ".xlsx":
        print(f"Sheet used: {sheet_name}")
    print(f"Rating/preference items ranked: {len(rankings)}")
    print(f"Saved: {OUTPUTS_DIR / 'rank_order.csv'}")
    print(f"Saved: {OUTPUTS_DIR / 'rank_order.png'}")


if __name__ == "__main__":
    main()
