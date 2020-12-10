# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/3
# @Author  : Yongrui Chen
# @File    : utils_esql.py
# @Software: PyCharm
"""
import re
import sys
import torch
sys.path.append("..")
from src.utils.utils import convert_val_index_from_cn_to_wp, convert_val_index_wp_to_string
from src.utils.utils import pred_sel_num, pred_sel_col, pred_sel_agg
from src.utils.utils import pred_where_join, pred_where_op, pred_where_num, pred_where_col, pred_where_agg, pred_where_val
from src.utils.utils import pred_ord, pred_ord_col, pred_ord_limit
from src.utils.utils import eval_col, eval_agg, eval_num, eval_val
from src.models.nn_utils import get_bert_emb, loss_esql, get_char_emb
from src.preprocess.enhance_header_esql import get_cn_toks


def get_fields_esql(one_batch, tables):
    q_batch = []
    q_tok_batch = []
    h_batch = []

    q_ch_batch = []
    h_ch_batch = []

    sel_batch = []
    agg_batch = []
    conds_join_batch = []
    conds_batch = []
    ord_batch = []

    sql_batch = []
    table_batch = []
    query_id_batch = []

    q_feature_batch = []

    for example in one_batch:
        example["question"] = example["question"].replace("（", "(").replace("）", ")")
        example["question_toks"] = [x.replace("（", "(").replace("）", ")") for x in example["question_toks"]]
        flag = True
        conds = example["sql"]["conds"]
        for cond in conds:
            if example["question"].find(cond[3]) == -1 \
                    or (cond[4] != None and example["question"].find(cond[4]) == -1):
                flag = False
        if not flag:
            continue

        q_batch.append(example["question"])
        q_tok_batch.append(example["question_toks"] + ["?"])
        h_batch.append(example["header"])

        q_ch_batch.append(list(example["question"]))
        h_ch_batch.append(example["header_aug"])

        sel_batch.append(example["sql"]["sel"])
        agg_batch.append(example["sql"]["agg"])
        conds_join_batch.append(example["sql"]["cond_conn_op"])
        conds_batch.append(example["sql"]["conds"])
        ord_batch.append(example["sql"]["ord_by"])

        query_id_batch.append(example["query_id"])
        sql_batch.append(example["sql"])
        table_batch.append(tables[example["table_id"]])

        q_feature_batch.append([x for x in example["question_feature"]])

    return q_batch, q_tok_batch, \
           h_batch, q_ch_batch, h_ch_batch, \
           sel_batch, agg_batch, conds_join_batch, conds_batch, ord_batch, \
           query_id_batch, sql_batch, table_batch, q_feature_batch

def get_fields_esql_for_prediction(one_batch, tables):
    q_batch = []
    q_tok_batch = []
    h_batch = []
    table_batch = []

    for example in one_batch:
        q_batch.append(example["question"])
        q_tok_batch.append(example["question_toks"] + ["?"])

        table = tables[example["table_id"]]
        table_batch.append(table)
        h_batch.append(table["header"])
    return q_batch, q_tok_batch, h_batch, table_batch

def get_ground_truth_esql(sel_batch, agg_batch, conds_join, conds_batch, ord_batch):
    gold_sel_num = []
    gold_sel_col = []
    gold_sel_agg = []

    for b, sel_col in enumerate(sel_batch):
        sel_agg = agg_batch[b]
        # convert format for unified processing
        gold_sel_num.append(len(sel_col))
        gold_sel_col.append(sel_col)
        gold_sel_agg.append(sel_agg)

    gold_where_join = [join for join in conds_join]

    gold_where_num = []
    gold_where_col = []
    gold_where_op = []
    gold_where_agg = []
    gold_where_val = []
    for b, conds in enumerate(conds_batch):
        gold_where_num.append(len(conds))
        gold_where_col.append([cond[2] for cond in conds])
        gold_where_agg.append([cond[0] for cond in conds])
        gold_where_op.append([cond[1] for cond in conds])
        gold_where_val.append([cond[3] for cond in conds])

    gold_ord = []
    gold_ord_col = []
    gold_ord_limit = []
    # for order by
    # 0: DESC, 1: ASC, 2: None
    for order_by in ord_batch:
        if order_by[0] == -1 or order_by[1] == 2:
            # [None, 2, None]
            gold_ord.append(2)
            gold_ord_col.append(0)
            gold_ord_limit.append(0)
        elif order_by[2] == -1:
            # [column_id, 0/1, None]
            gold_ord.append(order_by[1])
            gold_ord_col.append(order_by[0])
            gold_ord_limit.append(0)
        else:
            # [column_id, 0/1, limit_num]
            gold_ord.append(order_by[1])
            gold_ord_col.append(order_by[0])
            gold_ord_limit.append(order_by[2])

    return gold_sel_num, gold_sel_col, gold_sel_agg, \
           gold_where_join, gold_where_num, \
           gold_where_col, gold_where_agg, gold_where_op, gold_where_val, \
           gold_ord, gold_ord_col, gold_ord_limit

def token_match(toks1, toks2):
    for i in range(max(1, len(toks1) - len(toks2) + 1)):
        flag = True
        j = 0
        for _ in range(len(toks2)):
            if toks1[i + j] != toks2[j]:
                flag = False
                break
            j += 1
        if flag:
            return i, i + j
    return -1, -1

def get_value_index_chinese_character(q_tok, gold_val):
    gold_val_index = []
    for b, val_list in enumerate(gold_val):
        tmp = []
        for val in val_list:
            val_toks = get_cn_toks(str(val))
            st, ed = token_match(q_tok[b], val_toks)
            if st == -1 or ed == -1:
                print(q_tok[b])
                print(val_toks)
                raise("ValueFoundError")
            tmp.append((st, ed))
        gold_val_index.append(tmp)
    return gold_val_index

def total_to_sub(q_tok, h, h_num_limit):
    sub_q_tok = []
    sub_h = []
    sub_num = []
    for b, headers in enumerate(h):
        cnt = 0
        sub = 0
        while cnt + h_num_limit < len(headers):
            sub_q_tok.append(q_tok[b])
            sub_h.append(headers[cnt:cnt + h_num_limit])
            cnt += h_num_limit
            sub += 1
        sub_q_tok.append(q_tok[b])
        sub_h.append(headers[cnt:cnt + len(headers)])
        sub += 1
        sub_num.append(sub)
    return sub_q_tok, sub_h, sub_num

def sub_to_total(sub_q_emb, sub_q_lens, sub_h_nums,
                 sub_q_tok_wp, sub_cn_to_wp_index, sub_wp_to_cn_index, sub_num):
    """
    :param sub_q_emb: [sub_bs, max_q_len, q_h_bert]
    :param sub_q_lens: [sub_bs]
    :param sub_h_nums: [sub_bs]
    :param sub_num: [bs]
    :return:
    """
    # [bs, max_q_len, d_h_bert]
    q_emb = torch.stack([x.mean(dim=0) for x in sub_q_emb.split(sub_num)])
    # [bs]
    q_lens = torch.stack([x[0] for x in torch.tensor(sub_q_lens).split(sub_num)]).tolist()
    # [bs]
    h_nums = torch.stack([x.sum(dim=0) for x in torch.tensor(sub_h_nums).split(sub_num)]).tolist()

    q_tok_wp = []
    cn_to_wp_index = []
    wp_to_cn_index = []
    cnt = 0
    for i in range(len(sub_num)):
        q_tok_wp.append(sub_q_tok_wp[cnt])
        cn_to_wp_index.append(sub_cn_to_wp_index[cnt])
        wp_to_cn_index.append(sub_wp_to_cn_index[cnt])
        cnt += sub_num[i]
    return q_emb, q_lens, h_nums, q_tok_wp, cn_to_wp_index, wp_to_cn_index

def train_step(batch, tables, model, bert_model, bert_config, bert_tokenizer,
               ch_tokenizer, max_seq_len, n_layers_bert_out, h_num_limit, device=-1):
    # Get fields
    # q         : natural language question
    # q_tok_cn  : tokenized question by Chinese Character
    # h         : header
    # sel       : sql["sel"]
    # agg       : sql["agg"]
    # conds_join: sql["conds_conn_op"]
    # conds     : sql["conds"]
    # ord       : sql["ord"]
    # query_id  : the input is sub-query, this is the origin ID of each sub-query.
    # gold_sql  : sql
    # table_id  : sql["table_id"]
    q, q_tok_cn, h, q_ch, h_ch,\
    sel, agg, conds_join, conds, ord, \
    query_id, gold_sql, table, q_feature = get_fields_esql(batch, tables)

    gold_sel_num, gold_sel_col, gold_sel_agg, \
    gold_where_join, gold_where_num, \
    gold_where_col, gold_where_agg, gold_where_op, gold_where_val, \
    gold_ord, gold_ord_col, gold_ord_limit = get_ground_truth_esql(sel, agg, conds_join, conds, ord)

    # Because the length of inputs is longer than BERT's limit,
    # Split one table into several sub-tables, one sample into several sub-samples
    sub_q_tok_cn, sub_h, sub_num = total_to_sub(q_tok_cn, h, h_num_limit)

    # sub_q_emb         : [sub_bs, max_q_len, d_h_bert]
    # h_emb             : [total_h_num, d_h_bert]
    # sub_q_lens        : [sub_bs]
    # h_lens            : [total_h_num]
    # sub_h_nums        : [sub_bs], the number of columns (headers) of the sub-tables.
    # sub_q_tok_wp      : sub-question tokenized by WordPiece
    # sub_cn_to_wp_index: Index mapping from CoreNlp to WordPiece
    # sub_wp_to_cn_index: Index mapping from WordPiece to CoreNlp
    sub_q_emb, sub_q_lens, \
    h_emb, h_lens, sub_h_nums, \
    sub_q_tok_wp, sub_cn_to_wp_index, sub_wp_to_cn_index = get_bert_emb(bert_config, bert_model, bert_tokenizer,
                                                            sub_q_tok_cn, sub_h,
                                                            max_seq_len=max_seq_len,
                                                            n_layers_bert_out_q=n_layers_bert_out,
                                                            n_layers_bert_out_h=n_layers_bert_out,
                                                            device=device)
    # Combine sub-outputs into total-ouputs
    # q_emb         : [bs, max_q_len, d_h_bert]
    # q_lens        : [bs]
    # h_nums        : [bs]
    # q_tok_wp      : question tokenized by WordPiece
    # cn_to_wp_index: Index mapping from CoreNlp to WordPiece
    # wp_to_cn_index: Index mapping from WordPiece to CoreNlp
    q_emb, q_lens, h_nums, \
    q_tok_wp, cn_to_wp_index, wp_to_cn_index = sub_to_total(sub_q_emb, sub_q_lens, sub_h_nums,
                                                            sub_q_tok_wp, sub_cn_to_wp_index,
                                                            sub_wp_to_cn_index, sub_num)

    q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch = get_char_emb(model, ch_tokenizer,
                                                            q_ch, h_ch, device=device)

    try:
        # find the index of the gold value in the question.
        gold_where_val_index_cn = get_value_index_chinese_character(q_tok_cn, gold_where_val)
        gold_where_val_index_wp = convert_val_index_from_cn_to_wp(cn_to_wp_index,
                                                                  gold_where_val_index_cn)
    except:
        # if the gold value is not in the question, continue.
        print("Can not find gold value in the question, so continue next batch.")
        return None

    # score
    score_sel_num, score_sel_col, score_sel_agg, \
    score_where_join, score_where_num, \
    score_where_col, score_where_agg, score_where_op, score_where_val, \
    score_ord, score_ord_col, score_ord_limit = model(q_emb, q_lens, h_emb, h_lens, h_nums,
                                                      q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                                      q_feature,
                                                      gold_sel_num=gold_sel_num,
                                                      gold_sel_col=gold_sel_col,
                                                      gold_where_num=gold_where_num,
                                                      gold_where_col=gold_where_col,
                                                      gold_where_op=gold_where_op)

    # Calculate loss & step
    loss = loss_esql(score_sel_num, score_sel_col, score_sel_agg, score_where_join, score_where_num, score_where_col,
                       score_where_agg, score_where_op, score_where_val, score_ord, score_ord_col, score_ord_limit,
                       gold_sel_num, gold_sel_col, gold_sel_agg, gold_where_join, gold_where_num, gold_where_col,
                       gold_where_agg, gold_where_op, gold_where_val_index_wp, gold_ord, gold_ord_col, gold_ord_limit)

    return loss

def test(data_loader, tables, model, bert_model, bert_config, bert_tokenizer, ch_tokenizer,
         max_seq_len, n_layers_bert_out, h_num_limit, EG=False, device=-1):
    model.eval()
    bert_model.eval()

    cnt = 0
    cnt_sn = 0
    cnt_sc = 0
    cnt_sa = 0
    cnt_wj = 0
    cnt_wn = 0
    cnt_wc = 0
    cnt_wa = 0
    cnt_wo = 0
    cnt_wv = 0
    cnt_oo = 0
    cnt_oc = 0
    cnt_ol = 0
    cnt_lf = 0

    results = []
    for iB, batch in enumerate(data_loader):
        cnt += len(batch)

        q, q_tok_cn, h, q_ch, h_ch,\
        sel, agg, conds_join, conds, ord_by, \
        query_id, gold_sql, table, q_feature = get_fields_esql(batch, tables)

        gold_sel_num, gold_sel_col, gold_sel_agg, \
        gold_where_join, gold_where_num, \
        gold_where_col, gold_where_agg, gold_where_op, gold_where_val, \
        gold_ord, gold_ord_col, gold_ord_limit = get_ground_truth_esql(sel, agg, conds_join, conds, ord_by)

        sub_q_tok_cn, sub_h, sub_num = total_to_sub(q_tok_cn, h, h_num_limit)

        sub_q_emb, sub_q_lens, \
        h_emb, h_lens, sub_h_nums, \
        sub_q_tok_wp, sub_cn_to_wp_index, sub_wp_to_cn_index = get_bert_emb(bert_config, bert_model, bert_tokenizer,
                                                                sub_q_tok_cn, sub_h,
                                                                max_seq_len=max_seq_len,
                                                                n_layers_bert_out_q=n_layers_bert_out,
                                                                n_layers_bert_out_h=n_layers_bert_out,
                                                                device=device)
        # combine to total sample
        q_emb, q_lens, h_nums, q_tok_wp, \
        cn_to_wp_index, wp_to_cn_index = sub_to_total(sub_q_emb, sub_q_lens, sub_h_nums,
                                                      sub_q_tok_wp, sub_cn_to_wp_index,
                                                      sub_wp_to_cn_index, sub_num)

        q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch = get_char_emb(model, ch_tokenizer,
                                                                q_ch, h_ch, device=device)

        # model specific part
        # score
        if not EG:
            # No Execution guided decoding
            score_sel_num, score_sel_col, score_sel_agg, \
            score_where_join, score_where_num, \
            score_where_col, score_where_agg, score_where_op, score_where_val, \
            score_ord, score_ord_col, score_ord_limit = model(q_emb, q_lens, h_emb, h_lens, h_nums,
                                                              q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                                              q_feature)

            # prediction
            sel_num, sel_col, sel_agg, \
            where_join, where_num, where_col, where_agg, where_op, where_val_index, \
            ord, ord_col, ord_limit = pred_sql_esql(score_sel_num, score_sel_col, score_sel_agg, score_where_join,
                                                      score_where_num, score_where_col, score_where_agg, score_where_op,
                                                      score_where_val, score_ord, score_ord_col, score_ord_limit)

            # where_val_cn: list of CoreNlp tokens
            # where_val_wp: list of WordPiece tokens
            where_val_cn, where_val_wp = convert_val_index_wp_to_string(wp_to_cn_index, where_val_index,
                                                                        q_tok_cn, q_tok_wp, ad_hoc=False)

            pred_sql = generate_sql_esql(sel_num, sel_col, sel_agg, where_join, where_num, where_col,
                                           where_agg, where_op, where_val_cn, ord, ord_col, ord_limit, table)

        else:
           pass

        cnt_sel_num, cnt_sel_col, cnt_sel_agg, cnt_where_join, cnt_where_num, \
        cnt_where_col, cnt_where_agg, cnt_where_op, cnt_where_val, \
        cnt_ord, cnt_ord_col, cnt_ord_limit, cnt_lf_one = eval_sql_esql(sel_num, sel_col, sel_agg, where_join, where_num,
                                                                      where_col, where_agg, where_op,
                                                                      ord, ord_col, ord_limit, pred_sql,
                                                                      gold_sel_num, gold_sel_col, gold_sel_agg,
                                                                      gold_where_join, gold_where_num, gold_where_col,
                                                                      gold_where_agg, gold_where_op, gold_ord,
                                                                      gold_ord_col, gold_ord_limit, gold_sql)

        for b, sql in enumerate(pred_sql):
            result = {}
            result["query"] = sql
            result["table_id"] = table[b]["id"]
            result["nlu"] = q[b]
            result["lf"] = cnt_lf_one[b]
            results.append(result)

        # For each module, add record for combining sub queries
        cnt_sn += sum(cnt_sel_num)
        cnt_sc += sum(cnt_sel_col)
        cnt_sa += sum(cnt_sel_agg)
        cnt_wj += sum(cnt_where_join)
        cnt_wn += sum(cnt_where_num)
        cnt_wc += sum(cnt_where_col)
        cnt_wa += sum(cnt_where_agg)
        cnt_wo += sum(cnt_where_op)
        cnt_wv += sum(cnt_where_val)
        cnt_oo += sum(cnt_ord)
        cnt_oc += sum(cnt_ord_col)
        cnt_ol += sum(cnt_ord_limit)
        cnt_lf += sum(cnt_lf_one)

    acc_sn = cnt_sn / cnt
    acc_sc = cnt_sc / cnt
    acc_sa = cnt_sa / cnt
    acc_wj = cnt_wj / cnt
    acc_wn = cnt_wn / cnt
    acc_wc = cnt_wc / cnt
    acc_wa = cnt_wa / cnt
    acc_wo = cnt_wo / cnt
    acc_wv = cnt_wv / cnt
    acc_oo = cnt_oo / cnt
    acc_oc = cnt_oc / cnt
    acc_ol = cnt_ol / cnt
    acc_lf = cnt_lf / cnt

    acc = [acc_sn, acc_sc, acc_sa, acc_wj, acc_wn, acc_wc,
           acc_wa, acc_wo, acc_wv, acc_oo, acc_oc, acc_ol, acc_lf]
    return acc, results

def pred_sql_esql(score_sel_num, score_sel_col, score_sel_agg, score_where_join, score_where_num, score_where_col,
                    score_where_agg, score_where_op, score_where_val, score_ord, score_ord_col, score_ord_limit):
    sel_num = pred_sel_num(score_sel_num)
    sel_col = pred_sel_col(sel_num, score_sel_col)
    sel_agg = pred_sel_agg(sel_num, score_sel_agg)

    where_join = pred_where_join(score_where_join)
    where_num = pred_where_num(score_where_num)
    where_col = pred_where_col(where_num, score_where_col)
    where_agg = pred_where_agg(where_num, score_where_agg)
    where_op = pred_where_op(where_num, score_where_op)
    where_val_index = pred_where_val(where_num, score_where_val)

    ord = pred_ord(score_ord)
    ord_col = pred_ord_col(score_ord_col, ord)
    ord_limit = pred_ord_limit(score_ord_limit, ord)
    return sel_num, sel_col, sel_agg, \
           where_join, where_num, where_col, where_agg, where_op, where_val_index, \
           ord, ord_col, ord_limit

def eval_sql_esql(sel_num, sel_col, sel_agg, where_join, where_num, where_col, where_agg, where_op,
                    ord, ord_col, ord_limit, sql,
                    gold_sel_num, gold_sel_col, gold_sel_agg, gold_where_join, gold_where_num, gold_where_col,
                    gold_where_agg, gold_where_op, gold_ord, gold_ord_col, gold_ord_limit, gold_sql):
    cnt_sel_num = eval_num(gold_sel_num, sel_num)
    cnt_sel_col = eval_col(gold_sel_col, sel_col)
    cnt_sel_agg = eval_agg(gold_sel_col, gold_sel_agg, sel_agg)
    cnt_where_join = eval_num(gold_where_join, where_join)
    cnt_where_num = eval_num(gold_where_num, where_num)
    cnt_where_col = eval_col(gold_where_col, where_col)
    cnt_where_agg = eval_agg(gold_where_col, gold_where_agg, where_agg)
    cnt_where_op = eval_agg(gold_where_col, gold_where_op, where_op)
    cnt_where_val = eval_val(gold_where_col, gold_sql, sql, 3)
    cnt_ord = eval_num(gold_ord, ord)
    cnt_ord_col = eval_num(gold_ord_col, ord_col)
    cnt_ord_limit = eval_num(gold_ord_limit, ord_limit)

    cnt_lf = []
    for csn, csc, csa, cwj, cwn, cwc, cwa, cwo, cwv, coo, coc, col in zip(cnt_sel_num, cnt_sel_col, cnt_sel_agg,
                                                                          cnt_where_join, cnt_where_num, cnt_where_col,
                                                                          cnt_where_agg, cnt_where_op, cnt_where_val,
                                                                          cnt_ord, cnt_ord_col, cnt_ord_limit):
        if csn and csc and csa and cwj and cwn and cwc and cwa and cwo and cwv and coo and coc and col:
            cnt_lf.append(1)
        else:
            cnt_lf.append(0)

    return cnt_sel_num, cnt_sel_col, cnt_sel_agg, cnt_where_join, cnt_where_num, \
           cnt_where_col, cnt_where_agg, cnt_where_op, cnt_where_val, \
           cnt_ord, cnt_ord_col, cnt_ord_limit, cnt_lf

def generate_sql_esql(sel_num, sel_col, sel_agg, where_join, where_num, where_col,
                         where_agg, where_op, where_val_cn, ord, ord_col, ord_limit, table):
    AGGS = ['NONE', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    OPS = ['BETWEEN', '=', '>', '<', '>=', '<=', '!=']
    WOPS = ['None','OR', 'AND']
    ORDS = ['DESC', 'ASC', 'NONE']
    pred_sql = []
    bs = len(sel_num)
    for b in range(bs):
        header = table[b]["header_en"]
        conds = []
        for i_wn in range(where_num[b]):
            cond = []
            cond.append(AGGS[where_agg[b][i_wn]])
            cond.append(OPS[where_op[b][i_wn]])
            cond.append(header[where_col[b][i_wn]])
            cond.append(merge_val_to_chinese(where_val_cn[b][i_wn]))
            conds.append(cond)
        ord_by = [header[ord_col[b]], ORDS[ord[b]], ord_limit[b]]
        pred_sql.append({'agg': [AGGS[agg] for agg in sel_agg[b]],
                         'sel': [header[col] for col in sel_col[b]],
                         'conds_conn_op': WOPS[where_join[b]],
                         'conds': conds,
                         "ord_by": ord_by})
    return pred_sql

def merge_val_to_chinese(where_val_cn):
    return "".join(where_val_cn)

def add_sub_record(cnt_record, qid, res):
    if qid not in cnt_record:
        cnt_record[qid] = [res]
    else:
        cnt_record[qid].append(res)

def count_record(cnt_record):
    n_correct = 0
    for id, res in cnt_record.items():
        if sum(res) == len(res):
            n_correct += 1
    return n_correct

def combine_query(sub_queries):
    total_query = {'agg': [], 'sel': [], 'conds_conn_op': [], 'conds': [], 'ord_by': [0, 2, 0]}
    for query in sub_queries:
        total_query['agg'] += query['agg']
        total_query['sel'] += query['sel']
        total_query['conds'] += query['conds']
        total_query['conds_conn_op'] = query['conds_conn_op']
        if query['ord_by'][1] != 2:
            total_query['ord_by'] = query['ord_by']
    return total_query
