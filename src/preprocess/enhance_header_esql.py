# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/12
# @Author  : Yongrui Chen
# @File    : enhance_header_esql.py
# @Software: PyCharm
"""

import re
import os
import sys
import json
import pickle
import numpy as np
sys.path.append("../..")
from src.utils.utils import load_table
from src.utils.dictionary import Dictionary
import src.bert.tokenization as tokenization


p = re.compile(' ')
pattern_num = re.compile('\d+\.?\d*')

def edit_distance(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

def longest_common_subsequence(s1, s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = np.zeros((len1 + 1, len2 + 1))

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i -1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[len1][len2]


def literal_exact_match(q_str, c_str):
    if q_str.find(c_str) != -1:
        return True
    return False

def literal_score_match(q_tok_wp, c_tok_wp, max_n):
    q_len = len(q_tok_wp)
    c_str = " ".join(c_tok_wp)
    c_str_len = len(c_str)
    max_score = -1
    st = -1
    ed = -1
    # enumerate n-gram Word-Piece tokens
    for n in range(len(c_tok_wp), 0, -1):
        for i in range(q_len):
            if i + n > q_len:
                break
            q_str = " ".join(q_tok_wp[i : i + n])
            q_str_len = len(q_str)
            lcs = longest_common_subsequence(q_str, c_str)
            assert q_str_len > 0 and c_str_len > 0
            score = lcs * 1.0 / q_str_len + lcs * 1.0 / c_str_len
            if score > max_score:
                max_score = score
                st = i
                ed = i + n
                if max_score == 2:
                    return max_score, st, ed
    return max_score, st, ed

def enhance_example(bert_tokenizer, q_tok_cn, h, rows, max_n, threshold):
    q_str_cn = "".join(q_tok_cn).lower()
    q_tok_wp = []
    for tok in q_tok_cn:
        sub_toks = bert_tokenizer.tokenize(tok.lower())
        for sub_tok in sub_toks:
            q_tok_wp.append(sub_tok)
    h_aug = []
    q_feature = [0 for _ in q_tok_wp]
    for i, h_one in enumerate(h):
        aug = None
        contents = list(set([row[i] for row in rows]))
        for content_ori in contents:
            content = str(content_ori).lower()
            if q_str_cn.find(p.sub('', content)) == -1:
                continue
            c_tok_wp = bert_tokenizer.tokenize(content)
            max_score, st, ed = literal_score_match(q_tok_wp, c_tok_wp, max_n)
            if ed < len(q_tok_wp) and (q_tok_wp[ed] == "年" or q_tok_wp[ed] == "月" or q_tok_wp[ed] == "日"):
                continue
            if max_score > threshold:
                aug = str(content_ori)
                for j in range(st, ed):
                    q_feature[j] = 1
        if aug:
            h_aug.append(aug)
        else:
            h_aug.append("#NONE#")
    return h_aug, q_feature

def enhance_dataset(bert_path, path, table_path, out_path, max_n=10, threshold=1.9):
    vocab_file = os.path.join(bert_path, f'vocab.txt')
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tables = load_table(table_path)
    fout = open(out_path, "w", encoding="utf-8")
    cnt = 0
    with open(path) as fin:
        for line in fin:
            if cnt % 1000 == 0:
                print(cnt)
            cnt += 1
            example = json.loads(line.strip())
            table = tables[example["table_id"]]
            example["question_toks"] = get_cn_toks(example["question"])
            h_aug, q_feature = enhance_example(bert_tokenizer,
                                               example["question_toks"],
                                               table["header"],
                                               table["rows"],
                                               max_n, threshold)
            example["header_aug"] = h_aug
            example["question_feature"] = q_feature
            json_str = json.dumps(example, ensure_ascii=False, default=json_default_type_checker)
            json_str += "\n"
            fout.writelines(json_str)
    fout.close()

def json_default_type_checker(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def get_cn_toks(q):
    q_tok_cn = []
    last = 0
    for f in pattern_num.finditer(q):
        st, ed = f.span()
        q_tok_cn.extend(q[last : st])
        q_tok_cn.append(q[st : ed])
        last = ed
    q_tok_cn.extend(q[last : len(q)])
    return q_tok_cn

def mk_vocab(path):
    vocab = Dictionary()
    vocab.add_unk_token("#UNK#")
    vocab.add_pad_token("#PAD#")
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            for c in list(example["question"].lower()):
                vocab.add(c)
            for aug in list(example["header_aug"]):
                for c in list(aug.lower()):
                    vocab.add(c)
    print(len(vocab))
    return vocab

if __name__ == '__main__':

    bert_path = "../../data/bert-base-chinese/"
    enhance_dataset(
        bert_path,
        "../../data/esql/train.jsonl",
        "../../data/esql/tables.jsonl",
        "../../data/esql/train_aug.jsonl"
    )
    enhance_dataset(
        bert_path,
        "../../data/esql/dev.jsonl",
        "../../data/esql/tables.jsonl",
        "../../data/esql/dev_aug.jsonl"
    )
    enhance_dataset(
        bert_path,
        "../../data/esql/test.jsonl",
        "../../data/esql/tables.jsonl",
        "../../data/esql/test_aug.jsonl"
    )
    enhance_dataset(
        bert_path,
        "../../data/esql/dev_zs.jsonl",
        "../../data/esql/tables.jsonl",
        "../../data/esql/dev_zs_aug.jsonl"
    )
    enhance_dataset(
        bert_path,
        "../../data/esql/test_zs.jsonl",
        "../../data/esql/tables.jsonl",
        "../../data/esql/test_zs_aug.jsonl"
    )

    vocab = mk_vocab("../../data/esql/train_aug.jsonl")
    pickle.dump(vocab, open("../../data/esql/char_vocab.pkl", "wb"))