# Code borrowed and modified from here: https://github.com/phueb/BabyBERTa/tree/master/huggingface_recommended
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from multiprocessing import Pool
from time import time as timer  # Rename time to avoid conflicts
from scipy.stats import spearmanr
import torch.nn.functional as F
import torch.multiprocessing as mp


# def get_perplexity(model, tokenizer, sentence):
#     tensor_input = tokenizer.encode(sentence, return_tensors='pt')
#     repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
#     mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
#     masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
#     labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
#     with torch.inference_mode():
#         loss = model(masked_input.cuda(), labels=labels.cuda()).loss
#     return np.exp(loss.item())

# def get_scores_on_paradigm(model, tokenizer, file_path):
#     with open(file_path) as f:
#         data = list(f)
#     acc = 0
#     for item in data:
#         line = json.loads(item)
#         good = line["sentence_good"]
#         bad = line["sentence_bad"]
#         good_score = get_perplexity(sentence=good, model=model, tokenizer=tokenizer)
#         bad_score = get_perplexity(sentence=bad, model=model, tokenizer=tokenizer)
#         if bad_score >= good_score:
#             acc += 1
#     acc = acc / len(data)
#     return acc

def calculate_cosine_similarity(word1, word2, model, tokenizer):
    # 単語をトークン化してエンコーディング
    input_ids_word1 = tokenizer.encode(word1, return_tensors='pt').to(model.device)
    input_ids_word2 = tokenizer.encode(word2, return_tensors='pt').to(model.device)
    # モデルに入力して特徴量を取得
    with torch.no_grad():
        outputs_word1 = model(input_ids_word1)
        outputs_word2 = model(input_ids_word2)
    # print(outputs_word1)
    # 特徴量のcos類似度を計算
    similarity = F.cosine_similarity(outputs_word1.logits.mean(dim=1),
                                     outputs_word2.logits.mean(dim=1),
                                     dim=0)
    return similarity.mean().item()

def read_sim_test(file_path='./SimLex-999/SimLex-999.txt'):
    tests = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tmp = line.strip().split("\t")
            if len(tmp) != 10:
                continue
            word_pair = (tmp[0].lower(), tmp[1].lower())
            tests[word_pair] = float(tmp[3])
    return tests

# def calc_sim(w1, w2):
#     return np.dot(w1, w2)/np.linalg.norm(w1)/np.linalg.norm(w2)

# def test_sim(word_emb, tests):
#     pool = Pool(20)
#     real_tests = {}
#     for word_pair in tests:
#         w1 = word_pair[0]
#         w2 = word_pair[1]
#         if w1 in word_emb and w2 in word_emb:
#             real_tests[word_pair] = tests[word_pair]
#     print(f'{len(real_tests)}/{len(tests)} actual test cases!')
#     t0 = time()
#     args = [(word_emb[word_pair[0]], word_emb[word_pair[1]]) for word_pair in real_tests.keys()]
#     res = pool.starmap(calc_sim, args)
#     truth = list(real_tests.values())
#     rho = spearmanr(truth, res)[0]
#     print(f'Spearman coefficient: {rho}')
#     return rho

def calc_sim(vector1, vector2):
    # Assuming vector1 and vector2 are torch tensors
    vector1 = vector1.squeeze()
    vector2 = vector2.squeeze()
    # Calculate cosine similarity
    similarity = F.cosine_similarity(vector1, vector2, dim=0)
    return similarity.item()

def test_sim(model, tokenizer, tests):
    device = next(model.parameters()).device  # Get the device of the model
    pool = Pool(20)
    real_tests = {}
    for word_pair in tests:
        w1 = word_pair[0]
        w2 = word_pair[1]
        # Assuming tokenizer.encode returns a tensor
        if w1 in tokenizer.get_vocab() and w2 in tokenizer.get_vocab():
            real_tests[word_pair] = tests[word_pair]
    print(f'{len(real_tests)}/{len(tests)} actual test cases!')

    t0 = timer()
    args = [
        (model(**tokenizer(word_pair[0], return_tensors="pt").to(device)).logits.mean(dim=1).detach().to(device),
         model(**tokenizer(word_pair[1], return_tensors="pt").to(device)).logits.mean(dim=1).detach().to(device))
        for word_pair in real_tests.keys()
    ]
    res = pool.starmap(calc_sim, args)
    truth = list(real_tests.values())
    rho = spearmanr(truth, res)[0]
    print(f'Spearman coefficient: {rho}')
    return rho


def main(model_path):
    # model_path = 'model/aochildes-japanese/checkpoint-40000'
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaForMaskedLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # # 例: "cat" と "dog" のcos類似度を計算
    # word1 = "cat"
    # word2 = "dog"
    # similarity_score = calculate_cosine_similarity(word1, word2, model=model, tokenizer=tokenizer)
    # print(f"Cosine Similarity between '{word1}' and '{word2}': {similarity_score}")

    tests = read_sim_test()
    test_sim(model, tokenizer, tests)

    # print(get_perplexity(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))
    # print(get_perplexity(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))
    # path = "tests/wh_vs_that_with_gap_long_distance.jsonl"
    # # paths = glob.glob("tests/*.jsonl")
    # # for path in paths:
    # acc = get_scores_on_paradigm(model, tokenizer, path)
    # print(path + " " + str(acc*100))

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="script to train mini roberta model")
    parser.add_argument("--model_path", required=True, help="path to the model")

    args = parser.parse_args()
    main(args.model_path)