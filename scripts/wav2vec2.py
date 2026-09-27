"""
Run: python run_wav2vec2.py --dataset mcd_data or python run_wav2vec2.py --dataset accented_data

"""
## imports
import argparse
import torch 
import pandas as pd
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils import load_dataset, compute_wer, compute_bertscore_batch, save_results, log_progress, audio_conversion

MODEL_NAME = "wav2vec2"
MODEL_ID = "facebook/wav2vec-large-960h"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wav2Vec2 inference + scoring on a dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mcd_data", "accented_data"],
        help="Which dataset to run inference on.",
    )
    return parser.parse_args()

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model, device

def transcription(audio_path, processor, model, device):
    audio_path = audio_conversion(audio_path)
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription


def main():
    args = parse_args()
    dataset_name = args.dataset
    df = load_dataset(dataset_name)
    total = len(df)
    print(f"Loaded {total} samples from {dataset_name}")
    # load model
    processor, model, device = load_model()
    # transcribe
    results = []
    hypotheses = []
    references = []
    for i, row in df.iterrows():
        try:
            hypothesis = transcription(row["audio_path"], processor, model, device)
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