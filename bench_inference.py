#!/usr/bin/env python3
"""
bench_inference.py

Mesure simple de latence par prédiction.
Charge le modèle une seule fois, génère des exemples aléatoires
si aucune donnée d'entrée n'est fournie, et enregistre un rapport JSON.

Usage:
  python bench_inference.py --model exported_model/model/model.pkl --n 200 --out reports/bench_baseline.json

Respecte la règle: solution simple et réutilise le modèle existant.
"""
import argparse
import json
import os
import pickle
import statistics
from time import perf_counter

import numpy as np


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_single(model, x):
    # model.predict usually accepts 2D arrays
    return model.predict(x)


def get_n_features(model, fallback=20):
    # Try common sklearn attributes
    if hasattr(model, "n_features_in_"):
        return int(getattr(model, "n_features_in_"))
    if hasattr(model, "feature_names_in_"):
        return int(len(getattr(model, "feature_names_in_")))
    # Some wrappers store inside estimator_
    if hasattr(model, "estimator_"):
        return get_n_features(model.estimator_, fallback)
    return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the pickled model file")
    parser.add_argument("--n", type=int, default=200, help="Number of single-request inferences to measure")
    parser.add_argument("--out", default="reports/bench_baseline.json", help="Output JSON report path")
    args = parser.parse_args()

    model = load_model(args.model)
    n_features = get_n_features(model)

    rng = np.random.RandomState(0)
    X = rng.rand(args.n, n_features).astype(float)

    # Warmup
    try:
        _ = model.predict(X[:5])
    except Exception:
        pass

    times_ms = []
    for i in range(args.n):
        x = X[i : i + 1]
        t0 = perf_counter()
        _ = infer_single(model, x)
        t1 = perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    report = {
        "model": args.model,
        "n": args.n,
        "n_features": n_features,
        "latency_ms": {
            "min": min(times_ms),
            "median": statistics.median(times_ms),
            "p95": sorted(times_ms)[int(0.95 * len(times_ms)) - 1],
            "p99": sorted(times_ms)[int(0.99 * len(times_ms)) - 1],
            "mean": statistics.mean(times_ms),
        },
        "raw_ms": times_ms,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Bench terminé: {args.n} itérations. Rapport sauvé -> {args.out}")


if __name__ == "__main__":
    main()
