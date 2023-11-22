# Code borrowed and modified from here: https://github.com/phueb/BabyBERTa/tree/master/huggingface_recommended
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
logging.basicConfig(level=logging.ERROR)
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import random
import argparse


def load_sentences_from_file(file_path: Path,
                             include_punctuation: bool = True,
                             allow_discard: bool = False,
                             ) -> List[str]:
    """
    load sentences for language modeling from text file
    """

    print(f'Loading {file_path}', flush=True)

    res = []
    num_too_small = 0
    
    with open(file_path, 'r') as line_by_line_file:
    # with file_path.open('r') as line_by_line_file:

        for sentence in line_by_line_file.readlines():

            if not sentence:  # during probing, parsing logic above may produce empty sentences
                continue

            sentence = sentence.rstrip('\n')

            # check  length
            if sentence.count(' ') < 3 - 1 and allow_discard:
                num_too_small += 1
                continue

            if not include_punctuation:
                sentence = sentence.rstrip('.')
                sentence = sentence.rstrip('!')
                sentence = sentence.rstrip('?')

            res.append(sentence)

    if num_too_small:
        print(f'WARNING: Skipped {num_too_small:,} sentences which are shorter than {3}.')

    return res

from itertools import islice

def make_sequences(sentences: List[str],
                   num_sentences_per_input: int,
                   ) -> List[str]:

    gen = (bs for bs in sentences)

    # combine multiple sentences into 1 sequence
    res = []
    while True:
        sentences_in_sequence: List[str] = list(islice(gen, 0, num_sentences_per_input))
        if not sentences_in_sequence:
            break
        sequence = ' '.join(sentences_in_sequence)
        res.append(sequence)

    print(f'Num total sequences={len(res):,}', flush=True)
    return res

def get_perplexity(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
    with torch.inference_mode():
        loss = model(masked_input.cuda(), labels=labels.cuda()).loss
    return np.exp(loss.item())

from datasets import Dataset, DatasetDict

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed, TrainingArguments
from transformers import BertTokenizer

def get_scores_on_paradigm(model, tokenizer, file_path):
    with open(file_path) as f:
        data = list(f)
    
    acc = 0
    for item in data:
        line = json.loads(item)
        good = line["sentence_good"]
        bad = line["sentence_bad"]
        good_score = get_perplexity(sentence=good, model=model, tokenizer=tokenizer)
        bad_score = get_perplexity(sentence=bad, model=model, tokenizer=tokenizer)
        if bad_score >= good_score:
            acc += 1
    
    acc = acc / len(data)
    return acc

def main():
    tokenizer = RobertaTokenizerFast.from_pretrained('model/aochildes-japanese')
    model = RobertaForMaskedLM.from_pretrained('model/aochildes-japanese')

    print(get_perplexity(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))
    print(get_perplexity(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))
    path = "tests/wh_vs_that_with_gap_long_distance.jsonl"
    # paths = glob.glob("tests/*.jsonl")
    # for path in paths:
    acc = get_scores_on_paradigm(model, tokenizer, path)
    print(path + " " + str(acc*100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to train mini roberta model")

    args = parser.parse_args()
    main()