## imports
import argparse
import os
import time
import pandas as pd
import cohere
from utils import load_dataset, compute_wer, compute_bertscore_batch, save_results, log_progress, audio_conversion


MODEL_NAME = "cohere_transcribe"
COHERE_MODEL_ID = "cohere-transcribe-03-2026"  
SLEEP_SECONDS = 15   # rate limit: 5 requests/minute

def parse_args():
    parser = argparse.ArgumentParser(description="Run Cohere Transcribe inference + scoring on a dataset.")
    parser.add_argument("--dataset", required=True, choices=["mcd_data", "accented_data"])
    return parser.parse_args()

def load_client():
    api_key = os.environ.get("CO_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "CO_API_KEY not set. Export it before running: export CO_API_KEY='your-key-here'"
        )
    return cohere.ClientV2(api_key=api_key)

def transcribe(audio_path, client):
    audio_path = audio_conversion(audio_path)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=COHERE_MODEL_ID,
            language="en",
            file=f,
        )
    return response.text.strip()

def main():
    args = parse_args()
    dataset_name = args.dataset

    df = load_dataset(dataset_name)
    total = len(df)
    print(f"Loaded {total} samples from {dataset_name}")

    client = load_client()

    results = []
    hypotheses = []
    references = []
    for i, row in df.iterrows():
        try:
            hypothesis = transcribe(row["audio_path"], client)
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
        time.sleep(SLEEP_SECONDS)

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