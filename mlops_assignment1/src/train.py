r"""
Usage examples (run from repo root with venv active)

  # Iris classification
  python src/train.py --dataset iris --model logistic
  python src/train.py --dataset iris --model randomforest

  # Housing regression
  python src/train.py --dataset housing --model linear
  python src/train.py --dataset housing --model dtree

  # Custom CSV or custom label column
  python src/train.py --dataset housing --model linear \
                      --data-path C:\data\my_housing.csv \
                      --label-column MedHouseVal
"""

import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------- #
# Fixed paths for the cleaned datasets                                        #
# --------------------------------------------------------------------------- #
DEFAULT_PATHS = {
    "iris":    "data/processed/iris_clean.csv",
    "housing": "data/processed/housing_clean.csv",
}

# Candidate label‑column names (checked in order)
LABEL_MAP = {
    "iris": [
        "species", "class", "target"              # add more if your file differs
    ],
    "housing": [
        "median_house_value", "median_house_price",
        "median_value", "MedHouseVal", "target"
    ],
}

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_dataset(dataset_name: str,
                 custom_path: str | None = None,
                 explicit_label: str | None = None):
    """Return X_train, X_test, y_train, y_test for the requested dataset."""
    file_path = custom_path or DEFAULT_PATHS[dataset_name]
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Identify the label column
    if explicit_label:
        if explicit_label not in df.columns:
            raise KeyError(
                f"Explicit label '{explicit_label}' not found in {file_path}."
            )
        label_col = explicit_label
    else:
        for candidate in LABEL_MAP[dataset_name]:
            if candidate in df.columns:
                label_col = candidate
                break
        else:
            raise KeyError(
                f"No label column found for '{dataset_name}'. "
                f"Tried {LABEL_MAP[dataset_name]} but none were present."
            )

    X = df.drop(label_col, axis=1)
    y = df[label_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def choose_model(model_name: str):
    """Instantiate the requested model with reasonable defaults."""
    match model_name:
        case "logistic":
            return LogisticRegression(max_iter=200)
        case "randomforest":
            return RandomForestClassifier(n_estimators=200, max_depth=8)
        case "linear":
            return LinearRegression()
        case "dtree":
            return DecisionTreeRegressor(max_depth=10)
    raise ValueError(f"Unsupported model name: {model_name}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["iris", "housing"],
                        help="Which dataset to use.")
    parser.add_argument("--model", required=True,
                        choices=["logistic", "randomforest", "linear", "dtree"],
                        help="Which model to train.")
    parser.add_argument("--data-path",
                        help="Optional path to a pre‑processed CSV. "
                             "If omitted, the default cleaned file is used.")
    parser.add_argument("--label-column",
                        help="Override the auto‑detected label column name.")
    args = parser.parse_args()

    # MLflow setup ----------------------------------------------------------
    mlflow.set_tracking_uri("file:./mlruns")        # local tracking
    mlflow.set_experiment(f"{args.dataset}-experiments")

    with mlflow.start_run(run_name=f"{args.model}_{args.dataset}"):
        X_train, X_test, y_train, y_test = load_dataset(
            dataset_name=args.dataset,
            custom_path=args.data_path,
            explicit_label=args.label_column,
        )
        model = choose_model(args.model)
        model.fit(X_train, y_train)

        # -------------------------- Metrics --------------------------------
        if args.dataset == "iris":                  # classification
            accuracy = accuracy_score(y_test, model.predict(X_test))
            mlflow.log_metric("accuracy", accuracy)
            primary_metric = accuracy
        else:                                       # regression
            y_pred = model.predict(X_test)
            try:
                # Works on scikit‑learn ≥ 0.18
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                # Fallback for very old scikit‑learn builds
                rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mlflow.log_metric("rmse", rmse)
            primary_metric = -rmse                  # lower RMSE is better

        # ----------------------- Params & Artifacts ------------------------
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"{args.dataset}_model",
        )

        print(f"Run logged successfully. Primary metric = {primary_metric:.4f}")
