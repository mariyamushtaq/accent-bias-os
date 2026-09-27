## imports
import argparse
import torch
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import load_dataset, compute_wer, compute_bertscore_batch, save_results, log_progress, audio_conversion

MODEL_NAME = "whisper_large_v3"
MODEL_ID = "openai/whisper-large-v3"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Whisper Large v3 inference + scoring on a dataset.")
    parser.add_argument("--dataset", required=True, choices=["mcd_data", "accented_data"])
    return parser.parse_args()


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    whisper_processor = AutoProcessor.from_pretrained(MODEL_ID)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
        generate_kwargs={"language": "en"},
    )
    return whisper_pipe


def transcribe(audio_path, asr_pipe):
    audio_path = audio_conversion(audio_path)
    result = asr_pipe(audio_path, return_timestamps=False)
    return result["text"].strip()


def main():
    args = parse_args()
    dataset_name = args.dataset

    df = load_dataset(dataset_name)
    total = len(df)
    print(f"Loaded {total} samples from {dataset_name}")

    asr_pipe = load_model()
    results = []
    hypotheses = []
    references = []
    for i, row in df.iterrows():
        try:
            hypothesis = transcribe(row["audio_path"], asr_pipe)
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