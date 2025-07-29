import pandas as pd
from sklearn.datasets import load_iris
import pathlib

def main():
    iris = load_iris(as_frame=True)
    df = pd.concat([iris.data, iris.target.rename("class")], axis=1)
    out_path = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "iris.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
