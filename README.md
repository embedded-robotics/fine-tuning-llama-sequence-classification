# Fine-tuning LLaMA to predict Diagnosis-Related group for Patients using Sequence Classification

This repository contains the code used for [DRG-LLaMA : Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients](https://arxiv.org/abs/2309.12625) and implementation instructions with minor changes incorporated for testing or live inference.


## Description

This code base fine-tunes LLaMA model with a sequence classification head to predict DRG code based on the discharge summaries of the hospitalized patients: \
1. The discharge summaries come from MIMIC-IV (https://physionet.org/content/mimiciv/)
2. Discharge summaries are pre-processed to extract `brief hospital course` which is used as the input text
3. Single label DRG related codes are then extracted using a combination of `data/DRG_34.csv`, `DRG34_Mapping.csv` and `id2label.csv`. These labels are then used as the output labels
4. Base LLaMA model is then loaded using HuggingFace checkpoint `baffo32/decapoda-research-llama-7b-hf` and fine-tuned using sequence classification head to predict labelled DRG codes based on `brief hospital course` from discharge summaries
5. Fine-tuned checkpoints are then used along with `Gradio` application to predict DRG code and corresponding text for a seamless user experience. `testing_drg_llama` can be used for live inference via Gradio application


## Implementation Details
Please refer to https://github.com/hanyin88/DRG-LLaMA for the detailed implementation.

### Local setup
Install dependencies. We used conda environment.
```
conda env create -f environment.yml
```
Activate conda environment.
```
conda activate DRG-LLaMA
```


### MIMIC-IV pre-processing

1) You must have obtained access to MIMIC-IV database: https://physionet.org/content/mimiciv/. 
2) Download "discharge.csv" and "drgcodes.csv" from MIMIC-IV and update "dc_summary_path" in `paths.json` to the file locations. We provided mapping rule file in the data folder ("my_mapping_path").
3) We provided "DRG_34.csv" in the data folder, which is the official DRG v34.0 codes (https://www.cms.gov/icd10m/version34-fullcode-cms/fullcode_cms/P0372.html). 
4) We provided "DRG34_Mapping.csv", which is a mapping rule to unify MS-DRGs over years to a single version -- MS-DRG v34.0. Details of the method can be found in Supplemental Method 1 of the paper.  
5) In your terminal, navigate to the project directory, then type the following commands:
```
python -m data.MIMIC_Preprocessing
```
The script will generate files in "train_set_path", "test_set_path" and "id2label_path". These will be used for single label DRGs prediction.


### Running the models
We provided llama_single.py and llama_two.py, which implement fine-tuning of LLaMA with LoRA for the single label and two-label approaches of DRGs prediction, respectively. We largely adopted the framework from https://github.com/tloen/alpaca-lora.

Example usaige:
```
python -m llama_single --base_model 'decapoda-research/llama-7b-hf' --model_size '7b'
```
Hyperparameters can be adjusted such as:
```
python -m llama_single \
    --base_model 'decapoda-research/llama-7b-hf' \
    --model_size '7b' \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --cutoff_len 1024 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
```