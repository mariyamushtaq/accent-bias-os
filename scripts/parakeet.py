"""
Run: python parakeet.py --dataset mcd_data or python parakeet.py --dataset accented_data
"""
## imports
import argparse
import torch
import pandas as pd
import nemo.collections.asr as nemo_asr
from utils import load_dataset, compute_wer, compute_bertscore_batch, save_results, log_progress, audio_conversion

MODEL_NAME = "parakeet_tdt_0_6b_v3"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Parakeet-TDT 0.6B v3 inference + scoring on a dataset.")
    parser.add_argument("--dataset", required=True, choices=["mcd_data", "accented_data"])
    return parser.parse_args()


def load_model():
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    return model


def transcribe(audio_path, model):
    audio_path = audio_conversion(audio_path)
    result = model.transcribe([audio_path])
    return result[0].text.strip()


def main():
    args = parse_args()
    dataset_name = args.dataset

    df = load_dataset(dataset_name)
    total = len(df)
    print(f"Loaded {total} samples from {dataset_name}")

    model = load_model()

    results = []
    hypotheses = []
    references = []
    for i, row in df.iterrows():
        try:
            hypothesis = transcribe(row["audio_path"], model)
            hypotheses.append(hypothesis)
            references.append(row["reference_text"])
            result_row = {
                "audio_path": row["audio_path"],
                "reference_text": row["reference_text"],
                "hypothesis_text": hypothesis,
                "wer": compute_wer(row["reference_text"], hypothesis),
                "group": row["group"],
                "participant": row["participant"],
                "stem": row["stem"],
                "source_dataset": row["source_dataset"],
            }
            results.append(result_row)
        except Exception as e:
            print(f"\nError processing {row['audio_path']}: {e}")

        log_progress(i + 1, total, MODEL_NAME)

    print("\nComputing BERTScore in batch...")
    bert_scores = compute_bertscore_batch(references, hypotheses)
    for idx, result_row in enumerate(results):
        result_row["bertscore_precision"] = bert_scores["bertscore_precision"][idx]
        result_row["bertscore_recall"] = bert_scores["bertscore_recall"][idx]
        result_row["bertscore_f1"] = bert_scores["bertscore_f1"][idx]

    df_results = pd.DataFrame(results)
    save_results(df_results, MODEL_NAME, dataset_name)


if __name__ == "__main__":
    main()