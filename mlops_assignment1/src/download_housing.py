"""
Download the California Housing dataset from scikit-learn
and save it as mlops_assignment1/data/raw/housing.csv

Run from repo root:
> python src\\download_housing.py
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


def main() -> None:
    # 1. Fetch as a Bunch object (includes .data and .target)
    housing = fetch_california_housing(as_frame=True)

    # 2. Merge features + target into one DataFrame
    df = housing.frame  # already includes target column "MedHouseVal"

    # 3. Build output path (repo_root/data/raw/housing.csv)
    repo_root = Path(__file__).resolve().parents[1]
    # out_file = repo_root / "mlops_assignment1" / "data" / "raw" / "housing.csv"
    out_file = repo_root / "data" / "raw" / "housing.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 4. Save
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df):,} rows → {out_file.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
