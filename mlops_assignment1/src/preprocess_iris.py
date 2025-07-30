import pandas as pd
import argparse
import os


def load_and_clean(input_path: str, output_path: str) -> None:
    """
    Read iris.csv, perform minimal cleaning, and write a cleaned copy.

    Parameters
    ----------
    input_path : str
        Path to the raw iris CSV.
    output_path : str
        Destination for the cleaned CSV.
    """
    df = pd.read_csv(input_path)

    # Basic preprocessing: drop rows with any nulls
    df.dropna(inplace=True)

    # (Optional) Rename columns, encode labels, etc.
    # e.g., df.columns = [col.lower() for col in df.columns]

    # Ensure output directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data preprocessed → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="mlops_assignment1/data/raw/iris.csv",
        help="Path to the raw iris CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mlops_assignment1/data/processed/iris_clean.csv",
        help="Path for the cleaned iris CSV",
    )
    args = parser.parse_args()

    load_and_clean(args.input, args.output)
