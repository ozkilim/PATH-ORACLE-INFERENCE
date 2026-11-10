#!/usr/bin/env python3
"""
PATH — WSI-only prognostic risk model (LUAD, progression-free interval).

Input : a directory of TITAN slide embeddings (.h5, 768-d) and an optional
        slide->case mapping CSV. Without a mapping, each slide is its own case.
Output: per-slide and per-case risk scores + fixed-cutoff risk groups.

Usage:
    python run_PATH.py --embeddings /path/to/titan_h5_dir --out predictions.csv
    python run_PATH.py --embeddings dir/ --wsi-csv mapping.csv --out predictions.csv

mapping.csv columns: slide_id, case_id   (slide_id = h5 filename without .h5)

Dependencies: numpy, pandas, h5py  (no ML framework needed at inference).
"""
import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "PATH_v1.pkl"


def load_embedding(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        for key in ("features", "embedding", "embeddings"):
            if key in f:
                return np.asarray(f[key]).ravel()
    raise KeyError(f"No embedding dataset found in {h5_path} (looked for features/embedding/embeddings)")


def predict(bundle: dict, X: np.ndarray) -> np.ndarray:
    """X: (n_slides, 768) raw TITAN embeddings -> (n_slides,) risk scores."""
    Z = X @ bundle["svd_components"].T                      # TruncatedSVD.transform
    Z = (Z - bundle["scaler_mean"]) / bundle["scaler_scale"]  # StandardScaler (frozen, TCGA train)
    # Ensemble: mean of per-fold Cox partial hazards exp((z - mu_f) . beta_f)
    scores = np.zeros(len(Z))
    for f in bundle["folds"]:
        scores += np.exp((Z - f["norm_mean"]) @ f["beta"])
    return scores / len(bundle["folds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True, help="Directory of TITAN .h5 slide embeddings")
    ap.add_argument("--wsi-csv", default=None, help="Optional CSV with slide_id,case_id columns")
    ap.add_argument("--out", default="PATH_predictions.csv")
    args = ap.parse_args()

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    CUTOFF = bundle["cutoff"]

    emb_dir = Path(args.embeddings)
    h5s = sorted(emb_dir.glob("*.h5"))
    if not h5s:
        sys.exit(f"No .h5 files found in {emb_dir}")

    if args.wsi_csv:
        mapping = pd.read_csv(args.wsi_csv)
        mapping["slide_id"] = mapping["slide_id"].astype(str).str.replace(r"\.h5$", "", regex=True)
        slide_to_case = dict(zip(mapping["slide_id"], mapping["case_id"]))
    else:
        slide_to_case = None

    rows, X = [], []
    for h5 in h5s:
        sid = h5.stem
        case = slide_to_case.get(sid, sid) if slide_to_case is not None else sid
        if slide_to_case is not None and sid not in slide_to_case:
            continue
        X.append(load_embedding(h5))
        rows.append({"slide_id": sid, "case_id": case})
    if not rows:
        sys.exit("No slides matched the mapping CSV.")

    df = pd.DataFrame(rows)
    df["risk_score"] = predict(bundle, np.vstack(X))

    case_df = df.groupby("case_id", as_index=False)["risk_score"].mean()
    case_df["risk_group"] = np.where(case_df["risk_score"] >= CUTOFF, "High", "Low")
    df["risk_group"] = np.where(df["risk_score"] >= CUTOFF, "High", "Low")

    df.to_csv(Path(args.out).with_suffix(".slides.csv"), index=False)
    case_df.to_csv(args.out, index=False)
    print(f"PATH v{bundle['version']}  |  fixed cutoff = {CUTOFF:.6f} (High if score >= cutoff)")
    print(f"{len(df)} slides -> {len(case_df)} cases  |  High risk: {(case_df['risk_group']=='High').sum()}")
    print(f"Wrote {args.out} (per-case) and {Path(args.out).with_suffix('.slides.csv')} (per-slide)")


if __name__ == "__main__":
    main()
