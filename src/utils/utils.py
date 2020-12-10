# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : utils.py
# @Software: PyCharm
"""

import os
import sys
import json
import math
import nltk
import torch
import random
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer as bt
from transformers import BertModel as bm
from transformers import BertConfig as bc
sys.path.append("..")
import src.bert.tokenization as tokenization
from src.bert.modeling import BertModel, BertConfig


def get_dataset_name(path):
    if "train" in path:
        return "train"
    elif "dev" in path:
        return "dev"
    else:
        return "test"

def load_bert_from_tf(BERT_PT_PATH):
    bert_config_file = os.path.join(BERT_PT_PATH, f'config.json')

    bert_tokenizer = bt.from_pretrained(BERT_PT_PATH)
    bert_model = bm.from_pretrained(BERT_PT_PATH)
    bert_config = bc.from_json_file(bert_config_file)
    return bert_model, bert_tokenizer, bert_config

def load_bert(BERT_PATH):
    config_file = os.path.join(BERT_PATH, f'config.json')
    vocab_file = os.path.join(BERT_PATH, f'vocab.txt')
    init_checkpoint = os.path.join(BERT_PATH, f'pytorch_model.bin')

    bert_config = BertConfig.from_json_file(config_file)
    bert_config.print_status()

    bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    bert_model = BertModel(bert_config)
    bert_model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))

    return bert_model, bert_tokenizer, bert_config

def load_optimizers(model, bert_model, lr, bert_lr):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=0)

    bert_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, bert_model.parameters()),
                                lr=bert_lr, weight_decay=0)
    return opt, bert_opt

def load_table(path):
    tables = {}
    with open(path) as fin:
        for line in fin:
            t = json.loads(line.strip())
            tables[t["id"]] = t
    return tables

def load_data(sql_path, table_path):
    tables = load_table(table_path)

    datas = []
    with open(sql_path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            table = tables[example["table_id"]]
            example["header"] = table["header"]
            example["types"] = table["types"]
            example["table"] = table
            datas.append(example)
    return datas, tables

def load_meta_datas(sql_path, table_path, n_way, k_shot, n_tasks):
    tables = load_table(table_path)

    datas = {}
    with open(sql_path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            table = tables[example["table_id"]]
            example["header"] = table["header"]
            example["types"] = table["types"]
            if example["table_id"] not in datas:
                datas[example["table_id"]] = [example]
            else:
                datas[example["table_id"]].append(example)

    meta_datas = []
    for i in range(n_tasks):
        spt = []
        tables_spt = random.sample(datas.keys(), n_way)
        for t in tables_spt:
            spt.extend(random.sample(datas[t], min(k_shot, len(datas[t]))))
        qry = []
        tables_qry = []
        while len(tables_qry) < n_way:
            t = random.sample(datas.keys(), 1)
            while t[0] in tables_spt:
                t = random.sample(datas.keys(), 1)
            tables_qry += t
        for t in tables_qry:
            qry.extend(random.sample(datas[t], min(k_shot, len(datas[t]))))

        meta_datas.append([spt, qry])

    return meta_datas, tables

def get_query_set(spt, datas, n_samples):
    qry = []
    spt_tables = set([x["table_id"] for x in spt])
    while len(qry) < n_samples:
        sample = datas[random.randint(0, len(datas) - 1)]
        while sample["table_id"] in spt_tables:
            sample = datas[random.randint(0, len(datas) - 1)]
        qry.append(sample)
    return qry

def reinforce_header(q_tok, h, types, rows):
    new_h = []
    for j, header in enumerate(h):
        if types[j] == "text":
            for i in range(len(rows)):
                row_toks = nltk.word_tokenize(rows[i][j])
                if kmp(q_tok, row_toks):
                    header = header + " " + rows[i][j]
                    break
        new_h.append(header)
    return new_h

def get_n_gram_match(toks1, toks2):
    if len(toks2) > len(toks1):
        return False
    for i in range(len(toks1) - len(toks2) + 1):
        flag = True
        for j in range(len(toks2)):
            if toks1[j + i] != toks2[j]:
                flag = False
        if flag:
            return True
    return False

def kmp(toks, sub_toks):
    if len(toks) == 0:
        return 0
    if len(sub_toks) == 0:
        return 0
    next = [-1]* len(sub_toks)
    if len(sub_toks) > 1:
        next[1] = 0
        i, j = 1, 0
        while i < len(sub_toks) - 1:
            if j == -1 or sub_toks[i] == sub_toks[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]
    m = s = 0
    while s < len(sub_toks) and m < len(toks):
        if s == -1 or toks[m] == sub_toks[s]:
            m += 1
            s += 1
        else:
            s = next[s]
    if s == len(sub_toks):
        return True
    return False

def convert_val_index_from_cn_to_wp(cn_to_wp_index, val_index_cn):
    """
    Generate SQuAD style start and end index of wv in nlu. Index is for of after WordPiece tokenization.
    Assumption: where_str always presents in the nlu.
    """
    val_index_wp = []
    for b, _ in enumerate(val_index_cn):
        cn_to_wp = cn_to_wp_index[b]
        val_index_wp_one = []
        for i_wn,  index in enumerate(val_index_cn[b]):

            st_wp, ed_wp = index

            st_wp_idx = cn_to_wp[st_wp]
            ed_wp_idx = cn_to_wp[ed_wp]

            val_index_wp_one.append([st_wp_idx, ed_wp_idx])

        val_index_wp.append(val_index_wp_one)
    return val_index_wp

def convert_val_index_wp_to_string(wp_to_cn_index, val_index_wp, q_tok_cn, q_tok_wp, ad_hoc=True):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    val_str_wp = [] # word-piece version
    val_str_cn = []
    for b, val_index_wp_one in enumerate(val_index_wp):
        val_str_wp_one = []
        val_str_cn_one = []
        wp_to_cn_index_one = wp_to_cn_index[b]
        q_tok_wp_one = q_tok_wp[b]
        q_tok_cn_one = q_tok_cn[b]

        for i_wn, index in enumerate(val_index_wp_one):
            st_wp, ed_wp = index
            if ad_hoc:
                # Ad-hoc modification of ed_wp to deal with wp-tokenization effect.
                # e.g.) to convert "butler cc (" ->"butler cc (ks)" (dev set 1st question).
                val_str_wp_one.append(q_tok_wp_one[st_wp:ed_wp + 1])
            else:
                val_str_wp_one.append(q_tok_wp_one[st_wp:ed_wp])

            st_cn = wp_to_cn_index_one[st_wp]
            ed_cn = wp_to_cn_index_one[ed_wp]
            if ad_hoc:
                val_str_cn_one.append(q_tok_cn_one[st_cn:ed_cn + 1])
            else:
                val_str_cn_one.append(q_tok_cn_one[st_cn:ed_cn])

        val_str_wp.append(val_str_wp_one)
        val_str_cn.append(val_str_cn_one)

    return val_str_cn, val_str_wp

def pred_sel_num(score_sel_num):
    pred_num = []
    for score in score_sel_num:
        pred_num.append(score.argmax().item())
    return pred_num

def pred_sel_col(now_sel_num, score_sel_col):
    pred_cols = []
    for b, num in enumerate(now_sel_num):
        score = score_sel_col[b]
        pred = np.argsort(-score.data.cpu().numpy())[:num]
        pred.sort() # sort by index
        pred_cols.append(list(pred))
    return pred_cols

def pred_sel_col_for_beam(score_sel_col, beam_size):
    pred_sel_col_beam = []
    for scores in score_sel_col:
        val, idxs = scores.topk(k=beam_size)
        pred_sel_col_beam.append(idxs.tolist())
    return pred_sel_col_beam

def pred_sel_agg(now_sel_num, score_sel_agg):
    pred_aggs = []
    for b, score in enumerate(score_sel_agg):
        preds = []
        for i in range(now_sel_num[b]):
            preds.append(score[i].argmax().item())
        pred_aggs.append(preds)
    return pred_aggs

def pred_where_join(score_where_join):
    pred_join = []
    for score in score_where_join:
        pred_join.append(score.argmax().item())
    return pred_join

def pred_where_num(score_where_num):
    pred_num = []
    for score in score_where_num:
        pred_num.append(score.argmax().item())
    return pred_num

def pred_where_col(now_where_num, score_where_col):
    pred_cols = []
    for b, num in enumerate(now_where_num):
        score = score_where_col[b]
        pred = np.argsort(-score.data.cpu().numpy())[:num]
        pred.sort() # sort by index
        pred_cols.append(list(pred))
    return pred_cols

def pred_where_op(now_where_num, score_where_op):
    pred_ops = []
    for b, scores in enumerate(score_where_op):
        num = now_where_num[b]
        if num == 0:
            pred_ops.append([])
        else:
            preds = []
            for i in range(num):
                preds.append(scores[i].argmax().item())
            pred_ops.append(preds)
    return pred_ops

def pred_where_agg(now_where_num, score_where_agg):
    pred_aggs = []
    for b, scores in enumerate(score_where_agg):
        num = now_where_num[b]
        if num == 0:
            pred_aggs.append([])
        else:
            preds = []
            for i in range(num):
                preds.append(scores[i].argmax().item())
            pred_aggs.append(preds)
    return pred_aggs

def pred_where_val(now_where_num, score_where_val):
    score_val_st, score_val_ed = score_where_val.split(1, dim=3)
    score_val_st = score_val_st.squeeze(-1)
    score_val_ed = score_val_ed.squeeze(-1)

    val_st = score_val_st.argmax(dim=-1)
    val_ed = score_val_ed.argmax(dim=-1)

    pred_index = []
    for b, num in enumerate(now_where_num):
        pred = []
        for i in range(num):
            st = val_st[b][i]
            ed = val_ed[b][i]
            # swap
            if st > ed:
                st, ed = st, ed
            pred.append((st.item(), ed.item()))
        pred_index.append(pred)
    return pred_index

def pred_where_val_beam(max_wn, s_wv, beam_size):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx


    output:
    pr_wvi_beam = [B, max_wn, n_pairs, 2]. 2 means [st, ed].
    prob_wvi_beam = [B, max_wn, n_pairs]
    """
    bS = s_wv.shape[0]

    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)  # [B, 4, mL, 2] -> [B, 4, mL, 1], [B, 4, mL, 1]

    s_wv_st = s_wv_st.squeeze(3) # [B, 4, mL, 1] -> [B, 4, mL]
    s_wv_ed = s_wv_ed.squeeze(3)

    prob_wv_st = F.softmax(s_wv_st, dim=-1).detach().to('cpu').numpy()
    prob_wv_ed = F.softmax(s_wv_ed, dim=-1).detach().to('cpu').numpy()

    k_logit = int(math.ceil(math.sqrt(beam_size)))
    n_pairs = k_logit**2
    assert n_pairs >= beam_size
    values_st, idxs_st = s_wv_st.topk(k_logit) # [B, 4, mL] -> [B, 4, k_logit]
    values_ed, idxs_ed = s_wv_ed.topk(k_logit) # [B, 4, mL] -> [B, 4, k_logit]

    # idxs = [B, k_logit, 2]
    # Generate all possible combination of st, ed indices & prob
    pr_wvi_beam = [] # [B, max_wn, k_logit**2 [st, ed] paris]
    prob_wvi_beam = np.zeros([bS, max_wn, n_pairs])
    for b in range(bS):
        pr_wvi_beam1 = []

        idxs_st1 = idxs_st[b]
        idxs_ed1 = idxs_ed[b]
        for i_wn in range(max_wn):
            idxs_st11 = idxs_st1[i_wn]
            idxs_ed11 = idxs_ed1[i_wn]

            pr_wvi_beam11 = []
            pair_idx = -1
            for i_k in range(k_logit):
                for j_k in range(k_logit):
                    pair_idx += 1
                    st = idxs_st11[i_k].item()
                    ed = idxs_ed11[j_k].item()
                    pr_wvi_beam11.append([st, ed])

                    p1 = prob_wv_st[b, i_wn, st]
                    p2 = prob_wv_ed[b, i_wn, ed]
                    prob_wvi_beam[b, i_wn, pair_idx] = p1*p2
            pr_wvi_beam1.append(pr_wvi_beam11)
        pr_wvi_beam.append(pr_wvi_beam1)
    # prob
    return pr_wvi_beam, prob_wvi_beam

def pred_ord(score_ord):
    pred_ord = []
    for ord in score_ord:
        pred_ord.append(ord.argmax().item())
    return pred_ord

def pred_ord_col(score_ord_col, ord):
    pred_ord_col = []
    for i, col in enumerate(score_ord_col):
        if ord[i] != 2:
            pred_ord_col.append(col.argmax().item())
        else:
            pred_ord_col.append(0)
    return pred_ord_col

def pred_ord_limit(score_ord_limit, ord):
    pred_ord_limit = []
    for i, obl in enumerate(score_ord_limit):
        if ord[i] != 2:
            pred_ord_limit.append(obl.argmax().item())
        else:
            pred_ord_limit.append(0)
    return pred_ord_limit

def topk_multi_dim(tensor, n_topk=1, batch_exist=True):

    if batch_exist:
        idxs = []
        for b, tensor1 in enumerate(tensor):
            idxs1 = []
            tensor1_1d = tensor1.reshape(-1)
            values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
            idxs_list = np.unravel_index(idxs_1d.cpu().numpy(), tensor1.shape)
            # (dim0, dim1, dim2, ...)

            # reconstruct
            for i_beam in range(n_topk):
                idxs11 = []
                for idxs_list1 in idxs_list:
                    idxs11.append(idxs_list1[i_beam])
                idxs1.append(idxs11)
            idxs.append(idxs1)

    else:
        tensor1 = tensor
        idxs1 = []
        tensor1_1d = tensor1.reshape(-1)
        values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
        idxs_list = np.unravel_index(idxs_1d.numpy(), tensor1.shape)
        # (dim0, dim1, dim2, ...)

        # reconstruct
        for i_beam in range(n_topk):
            idxs11 = []
            for idxs_list1 in idxs_list:
                idxs11.append(idxs_list1[i_beam])
            idxs1.append(idxs11)
        idxs = idxs1
    return idxs

def remap_sc_idx(idxs, pr_sc_beam):
    for b, idxs1 in enumerate(idxs):
        for i_beam, idxs11 in enumerate(idxs1):
            sc_beam_idx = idxs[b][i_beam][0]
            sc_idx = pr_sc_beam[b][sc_beam_idx]
            idxs[b][i_beam][0] = sc_idx

    return idxs

def check_sc_sa_pairs(tb, pr_sc, pr_sa, ):
    """
    Check whether pr_sc, pr_sa are allowed pairs or not.
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    """
    bS = len(pr_sc)
    check = [False] * bS
    for b, pr_sc1 in enumerate(pr_sc):
        pr_sa1 = pr_sa[b]
        hd_types1 = tb[b]['types']
        hd_types11 = hd_types1[pr_sc1]
        if hd_types11 == 'text':
            if pr_sa1 == 0 or pr_sa1 == 3: # ''
                check[b] = True
            else:
                check[b] = False

        elif hd_types11 == 'real':
            check[b] = True
        else:
            raise Exception("New TYPE!!")

    return check

def eval_col(gold_col, pred_col):
    cnt_list= []
    for b, gold in enumerate(gold_col):

        pred = pred_col[b]
        pred_num = len(pred)
        gold_num = len(gold)

        if pred_num != gold_num:
            cnt_list.append(0)
            continue
        else:
            g = np.array(gold)
            g.sort()

            if np.array_equal(pred, g):
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list

def eval_agg(gold_col, gold_agg, pred_agg):
    """ pr's are all sorted as pred_sel_col are sorted in increasing order (in column idx)
        However, g's are not sorted.

        Sort g's in increasing order (in column idx)
    """
    cnt_list = []
    for b, g_agg in enumerate(gold_agg):
        g_col = gold_col[b]
        p_agg = pred_agg[b]
        p_num = len(pred_agg[b])
        g_num = len(gold_agg[b])

        if g_num != p_num:
            cnt_list.append(0)
            continue
        else:
            # Sort based wc sequence.
            idx = np.argsort(np.array(g_col))

            g_agg_s = np.array(g_agg)[idx]
            g_agg_s = list(g_agg_s)

            if type(p_agg) != list:
                raise TypeError
            if g_agg_s == p_agg:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list

def eval_num(gold_num, pred_num):
    cnt_list = []
    for b, gold in enumerate(gold_num):
        pred = pred_num[b]
        if pred == gold:
            cnt_list.append(1)
        else:
            cnt_list.append(0)
    return cnt_list

def eval_val(gold_col, gold_sql, pred_sql, val_id):
    """
    compare value string
    :param val_id: value position in sql["conds"], e.g., In WikiSQL, id is 2.
    :return:
    """
    cnt_list =[]
    for b, g_col in enumerate(gold_col):
        p_num = len(pred_sql[b]["conds"])
        g_num = len(gold_sql[b]["conds"])

        # Now sorting.
        # Sort based wc sequence.
        idx1 = np.argsort(np.array(g_col))

        if g_num != p_num:
            cnt_list.append(0)
            continue
        else:
            flag = True
            for i_wn, idx11 in enumerate(idx1):
                g = str(gold_sql[b]["conds"][idx11][val_id]).lower()
                p = str(pred_sql[b]["conds"][i_wn][val_id]).lower()
                if g != p:
                    flag = False
                    break
            if flag:
                cnt_list.append(1)
            else:
                cnt_list.append(0)
    return cnt_list

def save_for_evaluation(path_save, results, dset_name):
    path_save_file = os.path.join(path_save, f'results_{dset_name}.jsonl')
    with open(path_save_file, 'w', encoding='utf-8') as f:
        for i, r1 in enumerate(results):
            json_str = json.dumps(r1, ensure_ascii=False, default=json_default_type_checker)
            json_str += '\n'
            f.writelines(json_str)

def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, np.int64): return int(o)
    raise TypeError