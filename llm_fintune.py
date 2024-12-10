import os
from huggingface_hub import hf_hub_download
import torch
from constant import DRIVE_PATH

HG_API = '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'

model_id = 'lmsys/fastchat-t5-3b-v1.0'
filenames = [
    'pytorch_model.bin',
    'config.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'generation_config.json',
    'added_tokens.json',
    'spiece.model'
]

for filename in filenames:
    dowloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        use_auth_token=HG_API
    )
    print(dowloaded_model_path)

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=-1, max_length=1000)
print('PREDICTING...')
print(pipeline('teach me linear algebra', max_length=1000))

