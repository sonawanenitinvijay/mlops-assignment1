import pandas as pd
import argparse
import os

def load_and_clean(input_path, output_path):
    df = pd.read_csv(input_path)

    # Basic preprocessing: drop nulls, rename columns if needed
    df.dropna(inplace=True)

    # Save cleaned version
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Data preprocessed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, default='mlops_assignment1/data/raw/housing.csv')
    parser.add_argument('--output', type=str, default='mlops_assignment1/data/processed/housing_clean.csv')

    args = parser.parse_args()

    load_and_clean(args.input, args.output)