#!/usr/bin/env python3
"""
PATH-ORACLE — multimodal (WSI + RNA) prognostic risk model
(LUAD, progression-free interval). Early-fusion Cox ensemble.

Inputs:
  --embeddings  directory of TITAN slide embeddings (.h5, 768-d)
  --wsi-csv     CSV: slide_id, case_id
  --rna         CSV: case_id + the 46 signature genes (harmonized expression)

Output: per-case risk score + fixed-cutoff risk group.

NOTE: this model applies per-modality z-scoring ACROSS THE INPUT COHORT before
the frozen train scaler (exactly as in training). Run it on a cohort
(>= ~20 cases), not on a single patient.

Dependencies: numpy, pandas, h5py.
"""
import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "PATH_ORACLE_v1.pkl"


def load_embedding(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        for key in ("features", "embedding", "embeddings"):
            if key in f:
                return np.asarray(f[key]).ravel()
    raise KeyError(f"No embedding dataset found in {h5_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--wsi-csv", required=True, help="CSV with slide_id,case_id")
    ap.add_argument("--rna", required=True, help="CSV with case_id + 46 signature genes")
    ap.add_argument("--out", default="PATH_ORACLE_predictions.csv")
    args = ap.parse_args()

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    CUTOFF = bundle["cutoff"]
    feature_cols = bundle["feature_cols"]

    # ---- WSI: slide-level TITAN -> SVD(18) --------------------------------
    mapping = pd.read_csv(args.wsi_csv)
    mapping["slide_id"] = mapping["slide_id"].astype(str).str.replace(r"\.h5$", "", regex=True)
    emb_dir = Path(args.embeddings)
    rows, X = [], []
    for _, r in mapping.iterrows():
        h5 = emb_dir / f"{r['slide_id']}.h5"
        if not h5.exists():
            continue
        X.append(load_embedding(h5))
        rows.append({"slide_id": r["slide_id"], "case_id": r["case_id"]})
    if not rows:
        sys.exit("No slides found for the mapping CSV.")
    wsi = pd.DataFrame(rows)
    svd_cols = [c for c in feature_cols if c.startswith("histopathology_")]
    Z = np.vstack(X) @ bundle["svd_components"].T
    wsi[svd_cols] = Z[:, : len(svd_cols)]

    # ---- RNA: 46 signature genes ------------------------------------------
    rna = pd.read_csv(args.rna)
    missing = [g for g in bundle["rna_genes"] if g not in rna.columns]
    if missing:
        sys.exit(f"RNA CSV is missing {len(missing)} signature genes, e.g. {missing[:5]}")
    rna = rna[["case_id"] + bundle["rna_genes"]].rename(
        columns={g: f"transcriptomics_{g}" for g in bundle["rna_genes"]})

    # ---- Early fusion: inner-join on case_id (rows = slides) ---------------
    fused = wsi.merge(rna, on="case_id", how="inner")
    fused = fused.dropna(subset=feature_cols)
    if fused.empty:
        sys.exit("No cases with both modalities present.")

    # ---- Per-modality z-score within THIS cohort (as in training) ----------
    for prefix in ("histopathology_", "transcriptomics_"):
        cols = [c for c in feature_cols if c.startswith(prefix)]
        mu, sd = fused[cols].mean(), fused[cols].std(ddof=0)
        sd = sd.replace(0, 1.0)
        fused[cols] = (fused[cols] - mu) / sd

    # ---- Frozen train scaler + Cox fold ensemble ---------------------------
    F = fused[feature_cols].to_numpy(dtype=float)
    F = (F - bundle["scaler_mean"]) / bundle["scaler_scale"]
    scores = np.zeros(len(F))
    for f in bundle["folds"]:
        scores += np.exp((F - f["norm_mean"]) @ f["beta"])
    fused["risk_score"] = scores / len(bundle["folds"])

    case_df = fused.groupby("case_id", as_index=False)["risk_score"].mean()
    case_df["risk_group"] = np.where(case_df["risk_score"] >= CUTOFF, "High", "Low")
    case_df.to_csv(args.out, index=False)
    print(f"PATH-ORACLE v{bundle['version']}  |  fixed cutoff = {CUTOFF:.6f} (High if score >= cutoff)")
    print(f"{len(fused)} slide-rows -> {len(case_df)} cases  |  High risk: {(case_df['risk_group']=='High').sum()}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
