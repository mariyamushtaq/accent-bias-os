"""
ASR-OS: aggregate results analysis
Loads per-model-per-dataset result CSVs, sanity-checks them, computes
overall and group (native/non-native / accented sub-group) aggregate
WER and BERTScore (precision, recall, F1).

Assumes one CSV per model per dataset (e.g. 8 models x 2 datasets = 16 files),
each with columns:
audio_path, reference_text, hypothesis_text, wer, group, participant,
stem, source_dataset, bertscore_precision, bertscore_recall, bertscore_f1
"""

import pandas as pd
import glob
import os


RESULTS_DIR = "../results"

COL_WER = "wer"
COL_BERTSCORE_P = "bertscore_precision"
COL_BERTSCORE_R = "bertscore_recall"
COL_BERTSCORE_F1 = "bertscore_f1"
COL_GROUP = "group"            # native/non-native or accented sub-group
COL_SOURCE_DATASET = "source_dataset"  

# Keywords to identify model name from filename (checked in order, case-insensitive).
# Add/adjust to match your actual filenames.
MODEL_NAME_KEYWORDS = [
    ("distil_whisper", "Distil-Whisper v3.5"),
    ("wav2vec2", "Wav2Vec2"),
    ("whisper_large_v3", "Whisper Large V3"),
    ("cohere_transcribe", "Cohere Transcribe"),
    ("qwen3_asr_06b", "Qwen3-ASR 0.6B"),
    ("qwen3_asr_17b", "Qwen3-ASR 1.7B"),
    ("canary_qwen", "Canary-Qwen 2.5B"),
    ("parakeet_tdt", "Parakeet-TDT 0.6B v3"),
]


def infer_model_name(fname):
    """Match filename against known model keywords; fall back to raw filename."""
    lower = fname.lower()
    for keyword, clean_name in MODEL_NAME_KEYWORDS:
        if keyword in lower:
            return clean_name
    return fname


def load_and_sanity_check(results_dir):
    """
    Load every *_results.csv, print shape/columns/missing values.
    Returns a dict keyed by (model_name, dataset_name) -> DataFrame,
    so a model run on two datasets doesn't silently overwrite itself.
    """
    csv_paths = glob.glob(os.path.join(results_dir, "**", "*_results.csv"), recursive=True)
    print(f"Found {len(csv_paths)} CSVs\n")

    dfs = {}
    for path in csv_paths:
        fname = os.path.basename(path)
        model_name = infer_model_name(fname)
        df = pd.read_csv(path)

        # Determine dataset name from the source_dataset column if present,
        # otherwise fall back to filename.
        if COL_SOURCE_DATASET in df.columns and df[COL_SOURCE_DATASET].nunique() == 1:
            dataset_name = df[COL_SOURCE_DATASET].iloc[0]
        elif COL_SOURCE_DATASET in df.columns:
            dataset_name = "mixed"  # file contains multiple datasets, flag it
        else:
            dataset_name = fname

        key = (model_name, dataset_name)
        if key in dfs:
            print(f"[warn] duplicate key {key} from {fname} — overwriting previous entry, check filenames/data")
        dfs[key] = df

        print(f"--- {model_name} | {dataset_name} ({fname}) ---")
        print(f"  rows: {len(df)}")
        print(f"  columns: {list(df.columns)}")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing):
            print(f"  missing values:\n{missing}")
        print()

    return dfs


def compute_overall_summary(dfs):
    """Mean/median WER and mean BERTScore (P/R/F1) per model x dataset."""
    rows = []
    for (model_name, dataset_name), df in dfs.items():
        rows.append({
            "model": model_name,
            "dataset": dataset_name,
            "n_samples": len(df),
            "mean_wer": df[COL_WER].mean(),
            "median_wer": df[COL_WER].median(),
            "mean_bertscore_precision": df[COL_BERTSCORE_P].mean(),
            "mean_bertscore_recall": df[COL_BERTSCORE_R].mean(),
            "mean_bertscore_f1": df[COL_BERTSCORE_F1].mean(),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "mean_wer"])


def compute_group_breakdown(dfs):
    """Mean WER/BERTScore split by 'group' column, per model x dataset."""
    rows = []
    score_cols = [COL_WER, COL_BERTSCORE_P, COL_BERTSCORE_R, COL_BERTSCORE_F1]
    for (model_name, dataset_name), df in dfs.items():
        if COL_GROUP not in df.columns:
            print(f"[warn] {model_name} | {dataset_name}: no '{COL_GROUP}' column found, skipping breakdown")
            continue
        grouped = df.groupby(COL_GROUP)[score_cols].mean()
        for group_val, row in grouped.iterrows():
            rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "group": group_val,
                "mean_wer": row[COL_WER],
                "mean_bertscore_precision": row[COL_BERTSCORE_P],
                "mean_bertscore_recall": row[COL_BERTSCORE_R],
                "mean_bertscore_f1": row[COL_BERTSCORE_F1],
            })

    return pd.DataFrame(rows)


def main():
    dfs = load_and_sanity_check(RESULTS_DIR)

    summary_df = compute_overall_summary(dfs)
    print("=== Overall aggregate summary (per model x dataset) ===")
    print(summary_df.to_string(index=False))
    print()

    breakdown_df = compute_group_breakdown(dfs)
    print("=== Group breakdown (native/non-native or accented sub-group) ===")
    print(breakdown_df.to_string(index=False))

    summary_df.to_csv(os.path.join(RESULTS_DIR, "aggregate_summary.csv"), index=False)
    breakdown_df.to_csv(os.path.join(RESULTS_DIR, "group_breakdown_summary.csv"), index=False)
    print("\nSaved aggregate_summary.csv and group_breakdown_summary.csv to", RESULTS_DIR)


if __name__ == "__main__":
    main()