"""
usage: included --model_size flag rather than two separate scripts, since 0.6B/1.7B only differ by checkpoint ID
"""

## imports
import argparse
import torch
import pandas as pd
from qwen_asr import Qwen3ASRModel
from utils import load_dataset, compute_wer, compute_bertscore_batch, save_results, log_progress, audio_conversion

MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-ASR-0.6B",
    "1.7B": "Qwen/Qwen3-ASR-1.7B",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR inference + scoring on a dataset.")
    parser.add_argument("--dataset", required=True, choices=["mcd_data", "accented_data"])
    parser.add_argument("--model_size", required=True, choices=["0.6B", "1.7B"])
    return parser.parse_args()

def load_model(model_size):
    model = Qwen3ASRModel.from_pretrained(
        MODEL_IDS[model_size],
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,   # one file at a time to matche our loop pattern
        max_new_tokens=256,
    )
    return model

def transcribe(audio_path, model):
    audio_path = audio_conversion(audio_path)
    results = model.transcribe(audio=audio_path, language="English")
    return results[0].text.strip()

def main():
    args = parse_args()
    dataset_name = args.dataset
    model_size = args.model_size
    model_name = f"qwen3_asr_{model_size.lower().replace('.', '')}"  

    df = load_dataset(dataset_name)
    total = len(df)
    print(f"Loaded {total} samples from {dataset_name}")

    model = load_model(model_size)
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

        log_progress(i + 1, total, model_name)

    print("\nComputing BERTScore in batch...")
    bert_scores = compute_bertscore_batch(references, hypotheses)
    for idx, result_row in enumerate(results):
        result_row["bertscore_precision"] = bert_scores["bertscore_precision"][idx]
        result_row["bertscore_recall"] = bert_scores["bertscore_recall"][idx]
        result_row["bertscore_f1"] = bert_scores["bertscore_f1"][idx]

    df_results = pd.DataFrame(results)
    save_results(df_results, model_name, dataset_name)


if __name__ == "__main__":
    main()