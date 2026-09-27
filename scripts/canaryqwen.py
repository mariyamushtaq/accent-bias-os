## imports
import argparse
import torch
import pandas as pd
from nemo.collections.speechlm2.models import SALM
from utils import load_dataset, compute_wer, compute_bertscore_batch, save_results, log_progress, audio_conversion


MODEL_NAME = "canary_qwen_2_5b"
MODEL_ID = "nvidia/canary-qwen-2.5b"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Canary-Qwen 2.5B inference + scoring on a dataset.")
    parser.add_argument("--dataset", required=True, choices=["mcd_data", "accented_data"])
    return parser.parse_args()


def load_model():
    device = torch.device("cuda")
    model = SALM.from_pretrained(MODEL_ID).bfloat16().eval().to(device)
    return model

def transcribe(audio_path, model):
    audio_path = audio_conversion(audio_path)
    prompt = [{
        "role": "user",
        "content": f"Transcribe the following: {model.audio_locator_tag}",
        "audio": [audio_path],
    }]
    # if error, drop out max_new_tokens=256 (open issue on github rn)
    answer_ids = model.generate(prompts=[prompt], max_new_tokens=256)
    hypothesis = model.tokenizer.ids_to_text(answer_ids[0].cpu())
    return hypothesis.strip()

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