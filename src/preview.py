from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def detect_dataset_file(data_dir: Path) -> Path:
    excel_files = sorted(data_dir.glob("*.xlsx"))
    csv_files = sorted(data_dir.glob("*.csv"))
    if excel_files:
        return excel_files[0]
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(f"No .xlsx or .csv file found in {data_dir}")


def main() -> None:
    dataset_path = detect_dataset_file(DATA_DIR)
    print(f"Dataset detected: {dataset_path}")

    if dataset_path.suffix.lower() == ".xlsx":
        workbook = pd.ExcelFile(dataset_path)
        print("Sheet names:", workbook.sheet_names)
        df = pd.read_excel(dataset_path, sheet_name=workbook.sheet_names[0])
    else:
        print("Sheet names: N/A (CSV file)")
        df = pd.read_csv(dataset_path)

    print("Column names:")
    for column in df.columns:
        print(f"- {column}")

    print("First 5 rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
