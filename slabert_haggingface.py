# -*- coding: utf-8 -*-
"""slabert-haggingface.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C-EpFvKs0C-8CyiwnP_ZpyUPYC9B36mq
"""

# !pip3 install numpy torch transformers

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("Aqureshi71/slabert")
model = AutoModelForMaskedLM.from_pretrained("Aqureshi71/slabert")
model = model.cuda()  # モデルをGPUに移動 ここのエラー処理必要かも

import numpy as np
import torch

def get_perplexity(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.cuda(), labels=labels.cuda()).loss
    return np.exp(loss.item())

print(get_perplexity(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))
print(get_perplexity(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))