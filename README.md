# ASR-OS

Accent bias in open-source ASR. Clinical speech vs regular conversational speech.

## What this repo is

Scripts that load a dataset, transcribe it, and score WER (jiwer) + BERTScore (roberta-large). One script per model, same output columns.

Two datasets:

- `mcd_data` — Mozilla Common Voice subset. 400 clips, 200 native / 200 non-native. General conversational speech.
- `accented_data` — lab recordings of the four scripted STEM passages. Grouped Native / Native MD / Accented / Accented MD.

8 models: Qwen3-ASR 0.6B and 1.7B, Whisper Large V3, Distil-Whisper v3.5, Cohere Transcribe, Parakeet-TDT 0.6B v3, Canary-Qwen 2.5B, Wav2Vec2.

The runs are in. Numbers and the native vs non-native check are in `writeup.md`.

## How to run

Needs ffmpeg on PATH (wav conversion). Deps are in `requirements.txt`.

From the repo root:

```bash
python scripts/whisperlv3.py --dataset mcd_data
python scripts/whisperlv3.py --dataset accented_data
```

Same `--dataset` flag on the other scripts in `scripts/`. Qwen takes a size:

```bash
python scripts/qwen3asr.py --dataset accented_data --model_size 1.7B
```

Cohere reads `CO_API_KEY` from the environment. Export it before `scripts/coheretranscribe.py`.

Aggregate (has to be run from `scripts/`, it looks for `../results`):

```bash
cd scripts && python aggregate_analysis.py
```
