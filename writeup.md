# Write-up

Question was whether these ASR models miss more on non-native English when the speech is clinical than when it is just regular conversation.

## What was scored

Two sets, same 8 models, same columns.

- `mcd_data`: Mozilla Common Voice subset, 400 clips, 200 native and 200 non-native. Short conversational sentences.
- `accented_data`: 60 recordings, 16 people (8 native, 8 non-native), each reading the scripted STEM passages. Also split MD / non-MD.

WER is jiwer, lowercased. BERTScore is roberta-large. Both are already stored per clip in the result CSVs. The numbers below are the mean of those per-clip scores.

The check for an accent gap is a two-sided permutation test, 10,000 shuffles. Main numbers are clip-level, which matches how Common Voice was balanced (200/200). The clinical set is the same people read several times, so those clips are not independent. For that set I also averaged WER per person and ran the same test on the 8 vs 8 speaker means.

## Overall

Mean WER, best to worst within each dataset.

| Model | Common Voice | Clinical passages |
| --- | ---: | ---: |
| Cohere Transcribe | 0.073 | 0.131 |
| Parakeet-TDT 0.6B v3 | 0.087 | 0.150 |
| Canary-Qwen 2.5B | 0.097 | 0.317 |
| Qwen3-ASR 1.7B | 0.099 | 0.091 |
| Whisper Large V3 | 0.110 | 0.097 |
| Distil-Whisper v3.5 | 0.113 | 0.129 |
| Qwen3-ASR 0.6B | 0.119 | 0.137 |
| Wav2Vec2 | 0.333 | 0.501 |

On Common Voice about half the clips are exact for the newer models (zero WER around 48–65%). Median WER is 0 for Cohere, Parakeet, Canary, and Qwen3-ASR 1.7B. Wav2Vec2 almost never hits zero (1%).

On the STEM passages nothing is exact. Qwen3-ASR 1.7B is the lowest mean WER (0.091), then Whisper Large V3 (0.097), then Distil-Whisper (0.129). Canary looks fine in conversation (0.097) and falls off on the passages (0.317). Conversational WER does not rank these models the same way the clinical set does.

BERTScore F1 sits around 0.98–0.99 on Common Voice and 0.95–0.98 on the passages for everything except Wav2Vec2 (0.88 and 0.80).

## Native vs non-native

Gap is non-native mean WER minus native mean WER. Positive means worse on non-native.

Common Voice, clip-level. 200 and 200.

| Model | Native | Non-native | Gap | p |
| --- | ---: | ---: | ---: | ---: |
| Cohere Transcribe | 0.077 | 0.069 | -0.008 | 0.66 |
| Parakeet-TDT 0.6B v3 | 0.090 | 0.084 | -0.005 | 0.70 |
| Qwen3-ASR 1.7B | 0.099 | 0.099 | 0.000 | 1.00 |
| Canary-Qwen 2.5B | 0.095 | 0.099 | 0.004 | 0.85 |
| Whisper Large V3 | 0.104 | 0.116 | 0.012 | 0.55 |
| Qwen3-ASR 0.6B | 0.110 | 0.128 | 0.018 | 0.37 |
| Distil-Whisper v3.5 | 0.103 | 0.123 | 0.020 | 0.29 |
| Wav2Vec2 | 0.296 | 0.370 | 0.074 | 0.002 |

The newer models do not separate native from non-native here. Wav2Vec2 does, and it is also the weakest model overall.

Clinical passages, clip-level. 32 native clips, 28 non-native.

| Model | Native | Non-native | Gap | p |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B | 0.074 | 0.111 | 0.038 | 0.025 |
| Whisper Large V3 | 0.076 | 0.120 | 0.044 | 0.003 |
| Cohere Transcribe | 0.107 | 0.158 | 0.051 | 0.004 |
| Qwen3-ASR 0.6B | 0.106 | 0.172 | 0.066 | 0.004 |
| Distil-Whisper v3.5 | 0.095 | 0.168 | 0.073 | <0.001 |
| Parakeet-TDT 0.6B v3 | 0.110 | 0.197 | 0.087 | 0.002 |
| Wav2Vec2 | 0.449 | 0.560 | 0.111 | 0.002 |
| Canary-Qwen 2.5B | 0.259 | 0.383 | 0.124 | 0.036 |

Every model is worse on the non-native readings. The best model also has the smallest gap.

Same comparison after averaging each person's clips first (8 native speakers, 8 non-native). Direction does not change. With only 16 people, the tests are weaker: Distil-Whisper (p=0.005), Whisper Large V3 (0.024), Wav2Vec2 (0.029), Qwen3-ASR 0.6B (0.039), and Parakeet (0.050) still separate. Qwen3-ASR 1.7B's speaker gap is 0.031 (p=0.10), Cohere 0.042 (p=0.09), Canary 0.094 (p=0.26). Small gap plus small n is why the best model does not clear 0.05 once you stop treating four STEMs as four people.

BERTScore moves the same way, just less. Clinical F1 is about 0.01–0.02 lower for non-native speakers. On Common Voice the F1 gap is about 0, except Wav2Vec2 (about 0.015, p<0.001). Qwen3-ASR 1.7B's clinical F1 gap is the smallest (0.007, p=0.08).

## Result tables

Means of the per-clip scores. Same numbers as the aggregate CSVs, rounded to 3 decimals.

### Accented clinical data

| Model | N | Mean WER | Median WER | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B | 60 | 0.091 | 0.073 | 0.980 | 0.979 | 0.980 |
| Whisper Large V3 | 60 | 0.097 | 0.079 | 0.978 | 0.978 | 0.978 |
| Distil-Whisper v3.5 | 60 | 0.129 | 0.111 | 0.968 | 0.969 | 0.968 |
| Cohere Transcribe | 60 | 0.131 | 0.113 | 0.969 | 0.971 | 0.970 |
| Qwen3-ASR 0.6B | 60 | 0.137 | 0.112 | 0.970 | 0.967 | 0.968 |
| Parakeet-TDT 0.6B v3 | 60 | 0.150 | 0.119 | 0.967 | 0.963 | 0.965 |
| Canary-Qwen 2.5B | 60 | 0.317 | 0.313 | 0.968 | 0.937 | 0.952 |
| Wav2Vec2 | 60 | 0.501 | 0.500 | 0.775 | 0.824 | 0.799 |

### Common Voice

| Model | N | Mean WER | Median WER | BERTScore P | BERTScore R | BERTScore F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cohere Transcribe | 400 | 0.073 | 0.000 | 0.992 | 0.991 | 0.992 |
| Parakeet-TDT 0.6B v3 | 400 | 0.087 | 0.000 | 0.988 | 0.986 | 0.987 |
| Canary-Qwen 2.5B | 400 | 0.097 | 0.000 | 0.990 | 0.987 | 0.989 |
| Qwen3-ASR 1.7B | 400 | 0.099 | 0.000 | 0.989 | 0.987 | 0.988 |
| Whisper Large V3 | 400 | 0.110 | 0.029 | 0.985 | 0.983 | 0.984 |
| Distil-Whisper v3.5 | 400 | 0.113 | 0.071 | 0.984 | 0.982 | 0.983 |
| Qwen3-ASR 0.6B | 400 | 0.119 | 0.071 | 0.985 | 0.983 | 0.984 |
| Wav2Vec2 | 400 | 0.333 | 0.273 | 0.858 | 0.903 | 0.880 |

### Mean WER by group

Accented clinical data. Straight means per group, not a test. Clip counts: Native MD 16, Native non-MD 16, Non-Native MD 11, Non-Native non-MD 17.

| Model | Native MD | Native non-MD | Non-Native MD | Non-Native non-MD |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-ASR 1.7B | 0.070 | 0.078 | 0.100 | 0.118 |
| Whisper Large V3 | 0.071 | 0.081 | 0.108 | 0.129 |
| Distil-Whisper v3.5 | 0.092 | 0.098 | 0.156 | 0.176 |
| Cohere Transcribe | 0.098 | 0.117 | 0.152 | 0.163 |
| Qwen3-ASR 0.6B | 0.080 | 0.132 | 0.176 | 0.169 |
| Parakeet-TDT 0.6B v3 | 0.096 | 0.123 | 0.167 | 0.215 |
| Canary-Qwen 2.5B | 0.187 | 0.331 | 0.299 | 0.437 |
| Wav2Vec2 | 0.454 | 0.444 | 0.570 | 0.554 |

Common Voice. 200 clips each.

| Model | Native | Non-native |
| --- | ---: | ---: |
| Cohere Transcribe | 0.077 | 0.069 |
| Parakeet-TDT 0.6B v3 | 0.090 | 0.084 |
| Canary-Qwen 2.5B | 0.095 | 0.099 |
| Qwen3-ASR 1.7B | 0.099 | 0.099 |
| Whisper Large V3 | 0.104 | 0.116 |
| Distil-Whisper v3.5 | 0.103 | 0.123 |
| Qwen3-ASR 0.6B | 0.110 | 0.128 |
| Wav2Vec2 | 0.296 | 0.370 |

## MD split

On the clinical set, native MD is the lowest-WER group for most models, and non-native non-MD is usually the highest. Example, Qwen3-ASR 1.7B: Native MD 0.070, Native non-MD 0.078, Non-Native MD 0.100, Non-Native non-MD 0.118. Whisper Large V3 is the same shape (0.071, 0.081, 0.108, 0.129). That split sits on top of the accent gap. The cell sizes are 11–17 clips, so it is a pattern in the table, not a separate claim.

## Take

The accent gap shows up on the clinical passages and not on the Common Voice subset, for the models that are actually usable. Qwen3-ASR 1.7B and Whisper Large V3 are the ones to keep: lowest error on the passages, and the smallest native/non-native difference. Wav2Vec2 is bad on both, and it is the one model that already gaps on conversational speech. Canary's conversational score does not survive the STEM vocabulary.
