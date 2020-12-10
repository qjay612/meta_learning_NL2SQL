# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : utils_wikisql.py
# @Software: PyCharm
"""
import os
import sys
from numpy import array, argsort
sys.path.append('..')
from src.utils.utils import convert_val_index_from_cn_to_wp, convert_val_index_wp_to_string
from src.utils.utils import pred_where_op, pred_where_num, pred_where_col, pred_where_val
from src.utils.utils import eval_col, eval_agg, eval_num, eval_val
from src.utils.dbengine import DBEngine
from src.models.nn_utils import get_bert_emb, loss_wikisql, get_char_emb


def get_fields_wikisql(one_batch, tables):
    q_batch = []
    q_tok_batch = []
    h_batch = []

    q_ch_batch = []
    h_ch_batch = []

    sel_batch = []
    agg_batch = []
    conds_batch = []

    sql_batch = []
    table_batch = []

    q_feature_batch = []

    for example in one_batch:
        q_batch.append(example["question"])
        q_tok_batch.append(example["question_tok"])
        h_batch.append(example["header"])

        q_ch_batch.append(list(example["question"]))
        h_ch_batch.append(example["header_aug"])

        sel_batch.append(example["sql"]["sel"])
        agg_batch.append(example["sql"]["agg"])
        conds_batch.append(example["sql"]["conds"])

        sql_batch.append(example["sql"])
        table_batch.append(tables[example["table_id"]])

        q_feature_batch.append([x for x in example["question_feature"]])

    return q_batch, q_tok_batch, \
           h_batch, q_ch_batch, h_ch_batch, \
           sel_batch, agg_batch, conds_batch, \
           sql_batch, table_batch, q_feature_batch

def get_ground_truth_wiksql(sel_batch, agg_batch, conds_batch):
    gold_sel_col = []
    gold_sel_agg = []

    for b, sel_col in enumerate(sel_batch):
        sel_agg = agg_batch[b]
        # convert format for unified processing
        gold_sel_col.append(sel_col)
        gold_sel_agg.append(sel_agg)

    gold_where_col_num = []
    gold_where_col = []
    gold_where_op = []
    gold_where_val = []
    for b, conds in enumerate(conds_batch):
        gold_where_col_num.append(len(conds))
        gold_where_col.append([cond[0] for cond in conds])
        gold_where_op.append([cond[1] for cond in conds])
        gold_where_val.append([str(cond[2]).lower() for cond in conds])

    return gold_sel_col, gold_sel_agg, \
           gold_where_col_num, gold_where_col, gold_where_op, gold_where_val

def get_gold_where_val_index_corenlp(one_batch):
    gold_val_index_corenlp = []
    for example in one_batch:
        gold_val_index_corenlp.append(example["wvi_corenlp"])
    return gold_val_index_corenlp

def train_step(batch, tables, model, bert_model, bert_config, bert_tokenizer,
               ch_tokenizer, n_layers_bert_out, device=-1):
    # Get fields
    # q        : natural language question
    # q_tok_cn : tokenized question by CoreNlp
    # h        : header
    # q_ch     : natural language question char-level
    # h_ch     : header char-level
    # sel_col  : sql["sel"]
    # sel_agg  : sql["agg"]
    # conds    : sql["conds"]
    # sql      : sql
    # table    : batch of table
    # q_feature: question feature of table content matching
    q, q_tok_cn, h, q_ch, h_ch, \
    sel, agg, conds, sql, table, q_feature = get_fields_wikisql(batch, tables)

    gold_sel_col, gold_sel_agg, \
    gold_where_num, gold_where_col, \
    gold_where_op, gold_where_val = get_ground_truth_wiksql(sel, agg, conds)

    # get ground truth where-value index under CoreNLP tokenization scheme. It's done already on trainset.
    gold_where_val_index_cn = get_gold_where_val_index_corenlp(batch)

    # q_emb         : natural language question embedding from BERT
    # h_emb         : header embedding from BERT
    # q_lens        : token lengths of each question
    # h_lens        : token lengths of each header
    # h_nums        : the number of columns (headers) of the tables.
    # q_tok_wp      : question tokenized by WordPiece
    # cn_to_wp_index: Index mapping from CoreNlp to WordPiece
    # wp_to_cn_index: Index mapping from WordPiece to CoreNlp
    q_emb, q_lens, \
    h_emb, h_lens, h_nums, \
    q_tok_wp, cn_to_wp_index, wp_to_cn_index = get_bert_emb(bert_config, bert_model,
                                                            bert_tokenizer,
                                                            q_tok_cn, h,
                                                            n_layers_bert_out_q=n_layers_bert_out,
                                                            n_layers_bert_out_h=n_layers_bert_out,
                                                            device=device)

    # q_emb_ch      : natural language question embedding char-level
    # h_emb_ch      : header embedding char-level
    # q_lens_ch     : token lengths of each question char-level
    # h_lens_ch     : token lengths of each header char-level
    q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch = get_char_emb(model, ch_tokenizer,
                                                            q_ch, h_ch, device=device)

    try:
        gold_where_val_index_wp = convert_val_index_from_cn_to_wp(cn_to_wp_index,
                                                                  gold_where_val_index_cn)
    except:
        # Exception happens when where-condition is not found in nlu_tt.
        # In this case, that train example is not used.
        # During test, that example considered as wrongly answered.
        # e.g. train: 32.
        return None

    # score
    score_sel_col, score_sel_agg, score_where_num, \
    score_where_col, score_where_op, score_where_val = model(q_emb, q_lens, h_emb, h_lens, h_nums,
                                                             q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                                             q_feature,
                                                             gold_sel_col=gold_sel_col,
                                                             gold_where_num=gold_where_num,
                                                             gold_where_col=gold_where_col,
                                                             gold_where_op=gold_where_op)
    # Calculate loss & step
    loss = loss_wikisql(score_sel_col, score_sel_agg, score_where_num,
                        score_where_col, score_where_op, score_where_val,
                        gold_sel_col, gold_sel_agg, gold_where_num,
                        gold_where_col, gold_where_op, gold_where_val_index_wp)

    return loss

def test(data_loader, tables, model, bert_model, bert_config, bert_tokenizer, ch_tokenizer,
         n_layers_bert_out, st_pos=0, EG=False, beam_size=4, path_db=None, device=-1, dset_name='dev'):
    model.eval()
    bert_model.eval()

    cnt = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_lf = 0

    results = []

    engine = DBEngine(os.path.join(path_db, f"{dset_name}.db"))

    for iB, batch in enumerate(data_loader):

        cnt += len(batch)
        if cnt < st_pos:
            continue

        q, q_tok_cn, h, q_ch, h_ch, \
        sel, agg, conds, gold_sql, table, q_feature = get_fields_wikisql(batch, tables)

        gold_sel_col, gold_sel_agg, \
        gold_where_num, gold_where_col, \
        gold_where_op, gold_where_val = get_ground_truth_wiksql(sel, agg, conds)

        q_emb, q_lens, \
        h_emb, h_lens, h_nums, \
        q_tok_wp, cn_to_wp_index, wp_to_cn_index = get_bert_emb(bert_config, bert_model, bert_tokenizer,
                                                                q_tok_cn, h,
                                                                n_layers_bert_out_q=n_layers_bert_out,
                                                                n_layers_bert_out_h=n_layers_bert_out,
                                                                device=device)
        # char-level
        q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch = get_char_emb(model, ch_tokenizer,
                                                                q_ch, h_ch, device=device)

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            score_sel_col, score_sel_agg, score_where_num, \
            score_where_col, score_where_op, score_where_val = model(q_emb, q_lens, h_emb, h_lens, h_nums,
                                                                     q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                                                     q_feature)

            # prediction
            sel_col, sel_agg, where_num, \
            where_col, where_op, where_val_index = pred_sql_wikisql(score_sel_col, score_sel_agg,
                                                                    score_where_num, score_where_col,
                                                                    score_where_op, score_where_val)

            # where_val_cn: list of CoreNlp tokens
            # where_val_wp: list of WordPiece tokens
            where_val_cn, where_val_wp = convert_val_index_wp_to_string(wp_to_cn_index, where_val_index,
                                                                        q_tok_cn, q_tok_wp)

            pred_sql = generate_sql_wikisql(sel_col, sel_agg, where_num,
                                            where_col, where_op, where_val_cn, q)

        else:
            prob_sca, prob_w, prob_wn_w, sel_col, \
            sel_agg, where_num, pred_sql = model.beam_forward(q_emb, q_lens, h_emb, h_lens, h_nums,
                                                              q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                                              engine, table, q_tok_cn, q_tok_wp,
                                                              wp_to_cn_index, q, beam_size=beam_size)
            # sort and generate
            where_col, where_op, where_val_str, pred_sql = sort_and_generate_sql_wikisql(pred_sql)

        cnt_sel_col, cnt_sel_agg, \
        cnt_where_num, cnt_where_col, \
        cnt_where_op, cnt_where_val, cnt_lf_one = eval_sql_wikisql(sel_col, sel_agg, where_num,
                                                                   where_col, where_op, pred_sql,
                                                                   gold_sel_col, gold_sel_agg, gold_where_num,
                                                                   gold_where_col, gold_where_op, gold_sql)
        for b, sql in enumerate(pred_sql):
            result = {}
            result["query"] = sql
            result["table_id"] = table[b]["id"]
            result["nlu"] = q[b]
            result["sc"] = cnt_sel_col[b]
            result["sa"] = cnt_sel_agg[b]
            result["wn"] = cnt_where_num[b]
            result["wc"] = cnt_where_col[b]
            result["wo"] = cnt_where_op[b]
            result["wv"] = cnt_where_val[b]
            result["lf"] = cnt_lf_one[b]
            results.append(result)
        # count
        cnt_sc += sum(cnt_sel_col)
        cnt_sa += sum(cnt_sel_agg)
        cnt_wn += sum(cnt_where_num)
        cnt_wc += sum(cnt_where_col)
        cnt_wo += sum(cnt_where_op)
        cnt_wv += sum(cnt_where_val)
        cnt_lf += sum(cnt_lf_one)

    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wo = cnt_wo / cnt
    acc_wv = cnt_wv / cnt
    acc_lf = cnt_lf / cnt

    acc = [acc_sc, acc_sa, acc_wn, acc_wc, acc_wo, acc_wv, acc_lf]
    return acc, results

def pred_sql_wikisql(score_sel_col, score_sel_agg, score_where_num,
                     score_where_col, score_where_op, score_where_val):
    bs = len(score_sel_col)

    sel_col = pred_where_num(score_sel_col)
    sel_agg = pred_where_num(score_sel_agg)
    where_num = pred_where_num(score_where_num)
    where_col = pred_where_col(where_num, score_where_col)
    where_op = pred_where_op(where_num, score_where_op)
    where_val_index = pred_where_val(where_num, score_where_val)

    return sel_col, sel_agg, where_num, where_col, where_op, where_val_index

def eval_sql_wikisql(sel_col, sel_agg, where_num, where_col, where_op, sql,
                     gold_sel_col, gold_sel_agg, gold_where_num, gold_where_col, gold_where_op, gold_sql):
    cnt_sel_col = eval_num(gold_sel_col, sel_col)
    cnt_sel_agg = eval_num(gold_sel_agg, sel_agg)
    cnt_where_num = eval_num(gold_where_num, where_num)
    cnt_where_col = eval_col(gold_where_col, where_col)
    cnt_where_op = eval_agg(gold_where_col, gold_where_op, where_op)
    cnt_where_val = eval_val(gold_where_col, gold_sql, sql, 2)

    cnt_lf = []
    for csc, csa, cwn, cwc, cwo, cwv in zip(cnt_sel_col, cnt_sel_agg, cnt_where_num,
                                            cnt_where_col, cnt_where_op, cnt_where_val):
        if csc and csa and cwn and cwc and cwo and cwv:
            cnt_lf.append(1)
        else:
            cnt_lf.append(0)

    return cnt_sel_col, cnt_sel_agg, cnt_where_num, cnt_where_col, cnt_where_op, cnt_where_val, cnt_lf

def generate_sql_wikisql(sel_col, sel_agg, where_num, where_col, where_op, where_val_cn, q):
    pred_sql = []
    for b, _ in enumerate(q):
        conds = []
        for i_wn in range(where_num[b]):
            conds1 = []
            conds1.append(where_col[b][i_wn])
            conds1.append(where_op[b][i_wn])
            conds1.append(merge_val_to_english(where_val_cn[b][i_wn], q[b]))
            conds.append(conds1)

        pred_sql.append({'agg': sel_agg[b], 'sel': sel_col[b], 'conds': conds})
    return pred_sql

def sort_and_generate_sql_wikisql(pred_sql):
    pred_wc = []
    pred_wo = []
    pred_wv = []
    for b, pred_sql1 in enumerate(pred_sql):
        conds1 = pred_sql1["conds"]
        pred_wc1 = []
        pred_wo1 = []
        pred_wv1 = []

        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pred_wc1.append( conds11[0])
            pred_wo1.append( conds11[1])
            pred_wv1.append( conds11[2])

        # sort based on pred_wc1
        idx = argsort(pred_wc1)
        pred_wc1 = array(pred_wc1)[idx].tolist()
        pred_wo1 = array(pred_wo1)[idx].tolist()
        pred_wv1 = array(pred_wv1)[idx].tolist()

        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pred_wc.append(pred_wc1)
        pred_wo.append(pred_wo1)
        pred_wv.append(pred_wv1)

        pred_sql1['conds'] = conds1_sorted

    return pred_wc, pred_wo, pred_wv, pred_sql

def merge_val_to_english(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(',
               '-RRB-': ')',
               '-LSB-': '[',
               '-RSB-': ']',
               '``': '"',
               '\'\'': '"',
               }
               # '--': '\u2013'} # this generate error for test 5661 case.
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        # if '' (empty string) of None, continue
        if not raw_w_token:
            continue

        # Change the special characters
        w_token = special.get(raw_w_token, raw_w_token)  # maybe necessary for some case?

        # check the double quote
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear

        # Check whether ret is empty. ret is selected where condition.
        if len(ret) == 0:
            pass
        # Check blank character.
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            # Pad ' ' if ret + ' ' is part of nlq.
            ret = ret + ' '

        elif len(ret) > 0 and ret + w_token in nlq:
            pass  # already in good form. Later, ret + w_token will performed.

        # Below for unnatural question I guess. Is it likely to appear?
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '  # pad blank line between next token when " because in this case, it is of closing apperas
                # for the case of opening, no blank line.

        elif w_token[0] not in alphabet:
            pass  # non alphabet one does not pad blank line.

        # when previous character is the special case.
        elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token

    return ret.strip()