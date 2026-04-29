#!/usr/bin/env python3
"""
eval_after_opt.py

Compare les prédictions entre le modèle original et un modèle optimisé (si fourni).
Calcule accuracy simple si des labels sont fournis en CSV (colonne 'y') ou AUC si probabilités.

Usage:
  python eval_after_opt.py --orig exported_model/model/model.pkl --opt exported_model/model/model_onnx.pkl --testdata tests/test_labels.csv --out reports/eval_compare.json

Le script reste minimal et tolérant : si pas de données de test, il compare seulement les prédictions.
"""
import argparse
import json
import pickle
import os

import numpy as np

from sklearn.metrics import accuracy_score


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--opt", required=False)
    parser.add_argument("--testdata", required=False, help="CSV with features and a 'y' column")
    parser.add_argument("--out", default="reports/eval_compare.json")
    args = parser.parse_args()

    orig = load_model(args.orig)
    opt = load_model(args.opt) if args.opt else None

    # try to load test data if provided
    X = None
    y = None
    if args.testdata and os.path.exists(args.testdata):
        import pandas as pd

        df = pd.read_csv(args.testdata)
        if "y" in df.columns:
            y = df["y"].values
            X = df.drop(columns=["y"]).values
        else:
            X = df.values

    report = {"orig": str(args.orig), "opt": str(args.opt) if args.opt else None}

    if X is None:
        # generate tiny random inputs for comparison
        n_features = getattr(orig, "n_features_in_", 20)
        rng = np.random.RandomState(2)
        X = rng.rand(50, int(n_features))

    pred_orig = orig.predict(X)
    report["orig_pred_sample"] = pred_orig[:10].tolist()

    if opt is not None:
        try:
            pred_opt = opt.predict(X)
            report["opt_pred_sample"] = pred_opt[:10].tolist()
            report["agree_fraction"] = float((pred_orig == pred_opt).mean())
        except Exception as e:
            report["opt_error"] = str(e)

    if y is not None:
        report["orig_acc"] = float(accuracy_score(y, pred_orig))
        if opt is not None and "opt_pred_sample" in report:
            report["opt_acc"] = float(accuracy_score(y, pred_opt))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Evaluation saved -> {args.out}")


if __name__ == "__main__":
    main()
