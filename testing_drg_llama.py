import gradio as gr

import torch
import os
import pandas as pd
from peft import PeftConfig, PeftModel
from transformers import LlamaTokenizer, LlamaForSequenceClassification

def greet(discharge_summary):
    input = tokenizer(discharge_summary, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = inference_model(**input)

    prediction_label = output.logits.argmax(dim=-1).item()
    del output
    del input
    
    predicted_drg_code = merged_df.iloc[prediction_label]['drg_34_code']
    predicted_drg_desc = merged_df.iloc[prediction_label]['Description']

    return [predicted_drg_code, predicted_drg_desc]

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(label="Enter Discharge Summary", lines=30)],
    outputs=[gr.Textbox(label="DRG Code", lines=1),
             gr.Textbox(label="DRG Description", lines=3)],
)

if __name__ == "__main__":
    gpu_device_num=4
    torch.cuda.set_device(gpu_device_num)
    torch.cuda.current_device()
    device = torch.device(f"cuda:{gpu_device_num}" if torch.cuda.is_available() else "cpu")

    id2label_pd = pd.read_csv("data/id2label.csv")
    drg_desc_pd = pd.read_csv("data/drg_34_dissection.csv")
    drg_desc_pd = drg_desc_pd.rename(columns={"DRG": "drg_34_code"})
    merged_df = id2label_pd.merge(drg_desc_pd, how='left', on="drg_34_code")
    merged_df = merged_df[['drg_34_code', 'label', 'Description']].set_index('label', drop=True)

    checkpoint_id = "experiments/7b-512-4-2e-05-right-April-14-14-52"
    
    config = PeftConfig.from_pretrained(checkpoint_id)
    
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path,
                                           model_max_length=512,
                                           cache_dir="/data/mn27889/.cache/huggingface")
    tokenizer.pad_token_id = 0
    
    inference_model = LlamaForSequenceClassification.from_pretrained(config.base_model_name_or_path,
                                                       num_labels=738,
                                                       load_in_8bit=True,
                                                       torch_dtype=torch.float16,
                                                       cache_dir="/data/mn27889/.cache/huggingface")
    
    inference_model = PeftModel.from_pretrained(inference_model, checkpoint_id)
    
    inference_model = inference_model.to(device)
    inference_model.eval()
    
    text = '''
    ___ yo male with history of EtOH abuse (withdrawal seizures) who 
    presents with exertional chest pain X 48 hrs and increasing 
    depression, suicidal ideation X 6 weeks resulting in pt not 
    taking his medications for the last 5 days.

    Chest Pain - Patient was a poor historian. It was unclear if the 
    chest pain was cardiac or of other etiology. Differential 
    diagnosis included cardiac (stable angina, hypertensive urgency) 
    vs. GERD vs. Psychiatric vs. Musculoskeletal. Cardiac enzymes 
    were negative X3. However, patient has a number of coronary 
    artery disease risk factors, including father with fatal MI at 
    ___ years old, personal medical history of uncontrolled 
    hypertension, obesity and alcohol abuse. Patient's last ECHO was 
    done in ___ which showed preserved cardiac function 
    (LVEF 60%) but no previous stress test. A nuclear stress test 
    was ordered for the second day of hospitalization and patient 
    started on Atorvastatin 20mg. Patient, though, refused the 
    stress test and ate breakfast the morning of his scheduled 
    stress, stating that he did not need the test. LFTs came back 
    with slightly elevated AST/ALT but Lipid Panel was within normal 
    limits. Atorvastatin was discontinued. On the third day of 
    admission, the recommended nuclear stress was re-addressed with 
    patient, who again refused. On the day of discharge, patient had 
    recurrent chest pain which, per his report, was similar to the 
    pain from the day of admission. Pt refused medical interventions 
    at that time, including an EKG or nuclear stress. It was felt, 
    at that time, that the patient was competent to make this 
    decision as he reiterated understanding of the risks of refusing 
    treatment, including myocardial infarction, stroke, other 
    serious cardiac events. It was felt that patient's chest pain 
    was likely due to antihypertensive non-compliance. 

    Hypertension - Patient elicited non-compliance with medications 
    X5
    days, likely longer. Hypertension also exacerbated by EtOH 
    abuse. He was restarted on Atenolol 50mg BID, Amlodipine 5mg 
    daily which he started refusing the second day of admission. 
    Blood pressure was mildly decreased with the Valium as part of 
    the CIWA protocol, which he routinely triggered during the day. 
    Blood pressure went from SBP130s on the day of admission (after 
    taking medications) to 180/dopplerable on the day of discharge. 
    Patient was able to state his understanding that non-compliance 
    with these medications could increase his risk for stroke and 
    was likely contributing to his chest pain as hypertensive 
    urgency. 

    EtOH abuse - Patient's last drink was ___ pm on ___. 
    Patient was intoxicated in the ED and starting to withdraw 
    (symptoms of anxiety, tremulousness). Has history of seizures ___ 
    year ago), denies hallucinations/delirium tremens. Patient was 
    started on Diazepam ___ PO per CIWA protocol (>=10, q4hrs). 
    On the day of discharge, he triggered CIWA with score of 14 that 
    morning but had not triggered over the 8 hours overnight. 
    Patient was given 1L banana bag the first evening but switched 
    to PO B12, folate, multivitamin per patient's request. 
    Psychiatry and Social Work were consulted. They recommended 
    Section 35 for the patient as he had failed multiple 
    rehabilitation attempts and had been hospitalized multiple times 
    for similar symptoms (suicidal ideation, depression, alcohol 
    withdrawal). It was felt that the patient's primary issue is his 
    alcohol abuse and that his elicitation of suicidal thoughts is 
    often for secondary gain (please refer to patient's Section 35 
    for full explanation). 

    Depression/Suicidal Ideation: Patient contracted for safety 
    throughout his stay. Throughout this admission, he elicited 
    passive suicidal ideation (die by heart attack, overdose, hit by 
    train etc.). Patient had a 1:1 sitter throughout this admission. 
    Patient was continued on Ambien 5mg before bed as needed for 
    insomnia. He was also started on Risperidal BID PRN per 
    Psychiatry recommendations, which was not taken throughout this 
    stay. 

    Diarrhea/Groin Rash: Patient c/o diarrhea during this hospital 
    admission which was not witnessed by nursing but improved with 
    Immodium. Patient also described a pruritic groin rash from the 
    gel used for ___ ultrasound. He deferred physician examination 
    of the area, however, with the understanding that without 
    allowing an MD to examine the rash, he would not be given 
    topical medication for it. 

    GERD: Stable per patient, who was continued on Pantoprazole 40mg 
    daily. 

    Chronic lower back pain: Stable. Patient was continued on 
    Neurontin (300mg at 8 am, 300mg at 2 pm, 900mg. He was also 
    continued on Ibuprofen 600mg every 8 hours as needed.

    Gout: Stable, previously affected right great toe. No 
    Indomethicin was started. 

    FEN: Regular, cardiac diet

    PPx: Subcutaneous heparin, bowel regimen (docusate/senna PRN),
    ibuprofen 600mg q8hrs PRN pain.

    Communication: Patient, HCP: ___ (brother)

    Code: Full (confirmed with patient)

    '''
    
    demo.launch()