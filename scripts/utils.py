""" 
This script does the following: 
1) Loads and processes the datasets
2) Compute scores (WER & BERT-F1)
3) Outputs results into results/ folder
4) Logs for progress

"""


## imports
import os
import pandas as pd 
import numpy as np 
from pathlib import Path
from jiwer import wer
from bert_score import score as bert_score
import subprocess
import re
import glob

RERUN_MODE = False
RERUN_TAG = "rerun_2026_07_21"

## STEM reference texts
STEM_REFERENCES = {
    "STEM1": "Based on the results from your endoscopy, we saw evidence of gastritis, specifically erythematous changes along the mucosa of the anterior antrum, indicating inflammation likely due to your NSAID use or possible Helicobacter pylori infection. We also found several small gastric polyps, which are generally benign, but we have sent biopsies to confirm their histopathology and rule out any dysplasia. Additionally, there was mild esophagitis seen in the lower esophagus, suggesting gastroesophageal reflux disease or GERD as a contributing factor to your symptoms. Thankfully the duodenum appeared unremarkable, with no evidence of celiac disease or ulcerations. We will review these findings in light of your symptoms and lab results, and I may recommend proton pump inhibitors or other acid-reducing medications to alleviate the inflammation we saw.",
    "STEM2": "The patient is a 62-year-old male with a history of poorly controlled type 2 diabetes mellitus, osteoporosis, hypothyroidism, hypertension, and hyperlipidemia, presenting with a 2-month history of progressive exertional dyspnea and intermittent chest discomfort. His past medical history is Stemsignificant for a prior myocardial infarction 10 years ago, for which he underwent percutaneous coronary intervention with stent placement in the left anterior descending artery. His current medications include metformin, lisinopril, and atorvastatin, though his hemoglobin A1c remains elevated at 9.4%, indicating suboptimal glycemic control. On physical examination, he has bilateral lower extremity edema and jugular venous distension with a potential pneumothorax, suggestive of right-sided heart failure. Recent laboratory workup revealed an elevated BNP and mildly impaired renal function with a creatinine level of 1.6 mg/dL, AST of 56, and Alt of 77. Given his cardiovascular risk profile and recent symptoms, he was referred for further evaluation with echocardiography and stress testing.", 
    "STEM3": "This patient is a 57-year-old male with a history of IGG4 disease, Hyperkalaemia, Abdominal aortic aneurysm, here for longstanding ulcerative colitis, diagnosed in his early 20s, now complicated by pancolonic involvement and recurrent steroid dependence despite prior trials of mesalamine, azathioprine, and infliximab status post sigmoidectomy with colon to colon anastomosis and modified J pouch placement. He presents with a recent exacerbation characterized by increased stool frequency, tenesmus, and rectal bleeding, with endoscopic findings showing continuous mucosal friability and erosions extending from the rectum to the hepatic flexure. Additionally, he has a history of primary sclerosing cholangitis diagnosed three years ago, evidenced by characteristic beading and stricturing on MRCP, and is under surveillance for cholangiocarcinoma. His extraintestinal manifestations include arthralgias and intermittent episcleritis. Lab work shows mild anemia with iron deficiency, elevated CRP, and stable LFTs, although his ALP remains chronically elevated. Given the extent of disease and steroid dependence, I am considering the addition of a JAK inhibitor as well as a possible colectomy if his symptoms remain refractory.",
    "STEM4": "The patient is a 68-year-old male with a history of COPD, likely secondary to a 40-pack-year smoking history, who presented initially with progressively worsening dyspnea on exertion and intermittent wheezing over the past month. Pulmonary function testing revealed a FEV1/FVC ratio of 0.55, consistent with obstructive lung disease, and a significant reduction in DLCO, suggesting concomitant emphysematous changes. He has a baseline requirement of 2L home oxygen via nasal cannula, though he's recently needed intermittent increases to 3L, particularly at night. A recent high-resolution CT showed bilateral upper lobe predominant emphysema with apical bullae and scattered areas of bronchiectasis. His medications include tiotropium and albuterol inhalers, given a history of frequent exacerbations. No history of pulmonary hypertension or right heart failure, although echocardiography showed mild right ventricular hypertrophy, likely attributable to chronic hypoxic pulmonary vasoconstriction."
}

ACCENTED_GROUP_FOLDERS = {
    "Native": "Native non-MD",
    "Native MD": "Native MD",
    "Accented": "Non-Native non-MD",
    "Accented MD": "Non-Native MD",
}

AUDIO_EXTS = (".mp4", ".m4a")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Data")
CONVERTED_AUDIO_DIR = os.path.join(PROJECT_ROOT, "Data", "_converted_wav")

## Loading Data 
mcd_csv_path = os.path.join(DATA_ROOT, "MCD-Subset", "mcd_subset_validated.csv")
mcd_clips_dir = os.path.join(DATA_ROOT, "MCD-Subset", "mcd_validated_clips")
accented_clips_dir = os.path.join(DATA_ROOT, "Accented-Bias-Data", "Accented recording repository")


### Accented Data
def load_accented_data():
    """
    Walks: accented_clips_dir / [group_folder] / [participant] / STEM{n}.[mp4|m4a]
    Returns rows with reference text pulled from STEM_REFERENCES.
    """
    rows = []
    stem_pattern = re.compile(r"STEM[\s_]*([1-4])", re.IGNORECASE)

    for group_folder, group_label in ACCENTED_GROUP_FOLDERS.items():
        group_path = os.path.join(accented_clips_dir, group_folder)
        if not os.path.isdir(group_path):
            print(f"Warning: expected group folder not found: {group_path}")
            continue

        for participant in os.listdir(group_path):
            participant_path = os.path.join(group_path, participant.strip())
            if not os.path.isdir(participant_path):
                continue

            for filename in os.listdir(participant_path):
                if not filename.lower().endswith(AUDIO_EXTS):
                    continue
                
                match = stem_pattern.search(filename)
                if not match:
                    print(f"Warning: couldn't parse STEM number from {filename}, skipping")
                    continue
                
                stem_id = f"STEM{match.group(1)}"
                rec_path = os.path.join(participant_path, filename)

                rows.append({
                    "audio_path": rec_path,
                    "reference_text": STEM_REFERENCES[stem_id],
                    "group": group_label,
                    "participant": participant.strip(),
                    "stem": stem_id,
                    "source_dataset": "accented_data",
                })

    return pd.DataFrame(rows)
    


### MCD Data 
def load_mcd_data(mcd_csv_path=mcd_csv_path, mcd_clips_dir=mcd_clips_dir):
    df = pd.read_csv(mcd_csv_path)

    df_out = pd.DataFrame({
        "audio_path": df["path"].apply(lambda p: os.path.join(mcd_clips_dir, p)),
        "reference_text": df["sentence"],
        "group": df["native"],                # already 'native' / 'non_native' string
        "participant": df["client_id"],
        "stem": None,                          # not applicable — no STEM structure in mcd_data
        "source_dataset": "mcd_data",
    })
    return df_out

### loader
def load_dataset(dataset_name):
    if dataset_name == "mcd_data":
        return load_mcd_data()
    elif dataset_name == "accented_data":
        return load_accented_data()
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

## Scoring functions

### BERT score
def compute_bertscore(reference, hypothesis):
    P, R, F1 = bert_score([hypothesis], [reference], lang="en", model_type="roberta-large")
    # extracting scalar values
    precision = P.item()
    recall = R.item()
    f1_score = F1.item()

    return {
    'bertscore_precision': precision,
    'bertscore_recall': recall,
    'bertscore_f1': f1_score
}
    

def compute_bertscore_batch(references, hypotheses):
    P, R, F1 = bert_score(hypotheses, references, lang="en", model_type="roberta-large")
    return {
        "bertscore_precision": P.tolist(),
        "bertscore_recall": R.tolist(),
        "bertscore_f1": F1.tolist(),
    }

def compute_wer(reference, hypothesis):
    return wer(reference.lower(), hypothesis.lower())

## Results

def get_results_dir(dataset_name):
    if RERUN_MODE and dataset_name == "accented_data":
        return os.path.join(PROJECT_ROOT, "results", f"{dataset_name}_{RERUN_TAG}")
    return os.path.join(PROJECT_ROOT, "results", dataset_name)

def save_results(df_results, model_name, dataset_name):
    output_dir = get_results_dir(dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_results.csv")
    df_results.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    


## audio conversions

def audio_conversion(audio_path):
    # ffmpeg → 16kHz mono WAV, hash-cached
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        return audio_path
    """
    Mirror the path relative to data/, eg.: data/accented_data/Accented recording folder/native/personA/STEM1.mp4 -> data/_converted_wav/accented_data/Accented recording folder/native/personA/STEM1.wa 
        data/_converted_wav/
        ├── mcd_data/
        │   └── clips/
        │       ├── common_voice_en_33000174.wav
        │       └── ...
        └── accented_data/
            └── Accented recording folder/
                ├── native/
                │   └── personA/
                │       ├── STEM1.wav
                │       ├── STEM2.wav
                │       └── ...
                ├── accented MD/
                │   └── personB/
                │       └── ...
                └── ...
    
    """
    rel_path = os.path.relpath(audio_path, DATA_ROOT)
    rel_wav_path = os.path.splitext(rel_path)[0] + ".wav"
    cached_wav_path = os.path.join(CONVERTED_AUDIO_DIR, rel_wav_path)
    if os.path.exists(cached_wav_path):
        return cached_wav_path
    os.makedirs(os.path.dirname(cached_wav_path), exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1",
            cached_wav_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {audio_path}:\n{result.stderr}")

    return cached_wav_path

## Logs 
def log_progress(i, total, model_name):
    print(f"[{model_name}] {i}/{total} done", end="\r")