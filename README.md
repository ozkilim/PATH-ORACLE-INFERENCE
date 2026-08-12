# PATH-ORACLE — inference

Trained weights, fixed risk cutoffs, and inference scripts for the two
prognostic models from the **PATH-ORACLE** paper (LUAD, progression-free
interval).

**PATH** (`run_PATH.py`): TITAN slide embeddings → risk.
**PATH-ORACLE** (`run_PATH_ORACLE.py`): TITAN embeddings + 46-gene RNA signature → risk.

```bash
pip install -r requirements.txt   # numpy, pandas, h5py — nothing else

# WSI-only
python run_PATH.py --embeddings /path/to/titan_h5_dir \
                   --wsi-csv slides.csv --out predictions.csv

# Multimodal
python run_PATH_ORACLE.py --embeddings /path/to/titan_h5_dir \
                          --wsi-csv slides.csv --rna rna.csv \
                          --out predictions.csv
```

Output: per-case `risk_score` and `risk_group` — **High** if
`risk_score >= cutoff`. The fixed cutoff ships inside each pickle
(`bundle["cutoff"]`): the pre-specified 67th percentile of the out-of-fold
training scores (TCGA-LUAD), locked before validation.

## Step 1 — make the TITAN embeddings

One 768-d [TITAN](https://huggingface.co/MahmoodLab/TITAN) embedding per H&E
slide, via [Trident](https://github.com/mahmoodlab/TRIDENT):

```bash
python run_batch_of_slides.py --task all --wsi_dir /path/to/slides \
       --job_dir out/ --patch_encoder conch_v15 --slide_encoder titan \
       --mag 10 --patch_size 512
```

Point `--embeddings` at the resulting `slide_features_titan/` directory
(`.h5` filename minus extension = `slide_id`).

## Step 2 — input files

- `slides.csv` — `slide_id,case_id` (optional for `run_PATH.py`; default one case per slide)
- `rna.csv` — `case_id` + the 46 signature genes
  (`python -c "import pickle;print(pickle.load(open('models/PATH_ORACLE_v1.pkl','rb'))['rna_genes'])"`),
  harmonized log-scale expression

PATH scores each slide independently. PATH-ORACLE z-scores each modality
across the input cohort (as in training) — run it on a cohort, not a single patient.

## The pickles

Plain dicts of numpy arrays: SVD components, frozen scaler, 10 Cox fold models
(risk = mean of per-fold partial hazards), cutoff, feature lists, and the full
training recipe. Trained on the frozen TCGA-LUAD cohort; running these scripts
on CPTAC-LUAD reproduces the training pipeline's validation predictions to
machine precision (r = 1.000000).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
