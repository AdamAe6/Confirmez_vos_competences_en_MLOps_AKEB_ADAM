#!/usr/bin/env python3
"""
profile_inference.py

Exécute un profilage simple (cProfile) d'une boucle d'inférence afin de produire
un fichier .prof qui peut être inspecté avec pstats/snakeviz.

Usage:
  python -m cProfile -o reports/infer.prof profile_inference.py --model exported_model/model/model.pkl --n 100

"""
import argparse
import pickle
from time import perf_counter


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    model = load_model(args.model)

    # simple workload: run n predictions
    import numpy as np

    n_features = getattr(model, "n_features_in_", 20)
    rng = np.random.RandomState(1)
    X = rng.rand(args.n, int(n_features))

    for i in range(args.n):
        _ = model.predict(X[i : i + 1])


if __name__ == "__main__":
    main()
