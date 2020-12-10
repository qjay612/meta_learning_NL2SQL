# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : nn_utils.py
# @Software: PyCharm
"""

import torch
import torch.nn as nn

def get_bert_input(bert_tokenizer, q_tok_wp, h=None):
    """
    when h is None, the input only consists of question tokens.
    """
    tokens = []
    segment_ids = []

    tokens.append("[CLS]")
    index_q_st = len(tokens)

    segment_ids.append(0)
    for tok in q_tok_wp:
        tokens.append(tok)
        segment_ids.append(0)
    index_q_ed = len(tokens)
    index_q = (index_q_st, index_q_ed)    # start and end position of question tokens

    tokens.append("[SEP]")
    segment_ids.append(0)

    if h:
        index_h = []  # start and end position of header tokens
        for i, h_one in enumerate(h):
            index_h_st = len(tokens)
            if isinstance(h_one, str):
                sub_toks = bert_tokenizer.tokenize(h_one)
            else:
                sub_toks = []
                for h_one_tok in h_one:
                    sub_toks += bert_tokenizer.tokenize(h_one_tok)
            tokens += sub_toks
            index_h_ed = len(tokens)
            index_h.append((index_h_st, index_h_ed))
            segment_ids += [1] * len(sub_toks)
            if i < len(h) - 1:
                tokens.append("[SEP]")
                segment_ids.append(0)
            elif i == len(h) - 1:
                tokens.append("[SEP]")
                segment_ids.append(1)

        return tokens, segment_ids, index_q, index_h
    else:
        return tokens, segment_ids, index_q

def get_bert_output(bert_model, bert_tokenizer, q_tok_cn, h=None, max_seq_len=222, device=-1):
    """
    Input is tokenized further by WordPiece (wp) and fed into BERT.

    :param q_tok_cn: question tokens by Stanford CoreNlp tokenizer (English), or character (Chinese)
    :param h: headers, which are not tokenized
    :param max_seq_len: max length for input to BERT

    """
    q_lens = []
    h_nums = []

    input_ids = []
    tokens = []
    segment_ids = []
    input_mask = []

    index_q = []        # index for retrieve the position of question vectors
    index_h = []        # index for retrieve the position of header vectors

    cn_to_wp_index = []     # mapping from Corenlp tokenization to WordPiece
    wp_to_cn_index = []     # mapping from WordPiece tokenization to WordPiece

    q_tok_wp = []      # question tokenized by WordPiece

    _max_seq_len = 0

    for b, q_tok_cn_one in enumerate(q_tok_cn):

        cn_to_wp_index_one = []
        wp_to_cn_index_one = []
        q_tok_wp_one = []
        # build mapping when tokenize question
        for i, tok in enumerate(q_tok_cn_one):
            cn_to_wp_index_one.append(len(q_tok_wp_one))
            sub_toks = bert_tokenizer.tokenize(tok)

            for sub_tok in sub_toks:
                wp_to_cn_index_one.append(i)
                q_tok_wp_one.append(sub_tok)

        q_tok_wp.append(q_tok_wp_one)
        cn_to_wp_index.append(cn_to_wp_index_one)
        wp_to_cn_index.append(wp_to_cn_index_one)

        q_lens.append(len(q_tok_wp_one))

        if h:
            h_one = h[b]
            h_nums.append(len(h[b]))
            tokens_one, segment_ids_one, index_q_one, index_h_one = get_bert_input(bert_tokenizer,
                                                                                   q_tok_wp_one, h_one)
            index_h.append(index_h_one)
        else:
            tokens_one, segment_ids_one, index_q_one = get_bert_input(bert_tokenizer, q_tok_wp_one)

        index_q.append(index_q_one)

        _max_seq_len = max(_max_seq_len, len(tokens_one))

        input_ids_one = bert_tokenizer.convert_tokens_to_ids(tokens_one)
        input_mask_one = [1] * len(input_ids_one)

        input_ids.append(input_ids_one)
        tokens.append(tokens_one)
        segment_ids.append(segment_ids_one)
        input_mask.append(input_mask_one)

    # padding to max length
    for b in range(len(q_tok_cn)):
        while len(input_ids[b]) < max_seq_len:
            input_ids[b].append(0)
            input_mask[b].append(0)
            segment_ids[b].append(0)

    # convert to tensor
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    input_mask_tensor = torch.tensor(input_mask, dtype=torch.long)
    segment_ids_tensor = torch.tensor(segment_ids, dtype=torch.long)

    if device != -1:
        input_ids_tensor = input_ids_tensor.to(device)
        input_mask_tensor = input_mask_tensor.to(device)
        segment_ids_tensor = segment_ids_tensor.to(device)

    all_layers_bert_enc, pooling_output = bert_model(input_ids=input_ids_tensor,
                                                   token_type_ids=segment_ids_tensor,
                                                   attention_mask=input_mask_tensor)
    if h:
        h_lens = []
        for index_h_one in index_h:
            for index_st, index_ed in index_h_one:
                h_lens.append(index_ed - index_st)

        return all_layers_bert_enc, pooling_output, tokens, index_q, index_h, \
               q_tok_wp, q_lens, h_lens, h_nums, cn_to_wp_index, wp_to_cn_index

    else:
        return all_layers_bert_enc, pooling_output, tokens, index_q, \
               q_tok_wp, q_lens, cn_to_wp_index, wp_to_cn_index

def get_question_emb(all_layers_bert_enc, index_q, q_lens, d_h_bert,
                     n_layers_bert_out, device=-1):
    """
    extract question embedding from bert output by index
    :param index_q: start position and end position of the question.
    :param d_h_bert: 768 or 1024
    :param n_layers_bert:  12 or 24
    :param n_out_layers: the number of layers used, e.g., the last 2 layers
    :return:
    """
    n_layers_bert = len(all_layers_bert_enc)
    bs = len(q_lens)
    max_q_len = max(q_lens)
    if device != -1:
        emb = torch.zeros([bs, max_q_len, d_h_bert * n_layers_bert_out]).to(device)
    else:
        emb = torch.zeros([bs, max_q_len, d_h_bert * n_layers_bert_out])

    for b in range(bs):
        index_q_one = index_q[b]
        for j in range(n_layers_bert_out):
            # index of last j-th layer
            i_layer = n_layers_bert - 1 - j
            st = j * d_h_bert
            ed = (j + 1) * d_h_bert
            emb[b, 0:(index_q_one[1] - index_q_one[0]), st:ed] = \
                all_layers_bert_enc[i_layer][b, index_q_one[0]:index_q_one[1], :]
    return emb

def get_header_emb(all_layers_bert_enc, index_h, h_lens, h_nums,
                   d_h_bert, n_layers_bert_out, device=-1):
    """
    extract question embedding from bert output by index
    :param index_h: start position and end position of the question.
    :param h_lens: the length of each header
    :param h_nums: the number of headers in each table
    :param d_h_bert: 768 or 1024
    :param n_layers_bert:  12 or 24
    :param n_out_layers: the number of layers used, e.g., the last 2 layers
    :return:
    """
    n_layers_bert = len(all_layers_bert_enc)
    total_h_num = sum(h_nums)
    max_h_len = max(h_lens)
    if device != -1:
        emb = torch.zeros([total_h_num, max_h_len, d_h_bert * n_layers_bert_out]).to(device)
    else:
        emb = torch.zeros([total_h_num, max_h_len, d_h_bert * n_layers_bert_out])

    h_id = -1
    for b, index_h_one in enumerate(index_h):
        for index_st, index_ed in index_h_one:
            h_id += 1
            for j in range(n_layers_bert_out):
                # index of last j-th layer
                i_layer = n_layers_bert - 1 - j
                st = j * d_h_bert
                ed = (j + 1) * d_h_bert
                emb[h_id, 0:(index_ed - index_st), st:ed] = \
                    all_layers_bert_enc[i_layer][b, index_st:index_ed, :]
    return emb

def get_bert_emb(bert_config, bert_model, bert_tokenizer, q_tok_cn, h=None, max_seq_len=222,
                 n_layers_bert_out_q=1, n_layers_bert_out_h=1, device=-1):
    """
    :param q_tok: question tokenized by CoreNlp
    :param h: headers
    :param n_layers_bert_out_q:  the last n layers of BERT used for question
    :param n_layers_bert_out_h:  the last n layers of BERT used for header
    :return:
    """
    if h:
        # get contextual output of all tokens from bert
        # all_layers_bert_enc: BERT outputs from all layers.
        # pooling_output: output of [CLS] vector.
        all_layers_bert_enc, pooling_output, \
        tokens, index_q, index_h, \
        q_tok_wp, q_lens, h_lens, h_nums, \
        cn_to_wp_index, wp_to_cn_index = get_bert_output(bert_model, bert_tokenizer,
                                                         q_tok_cn, h, max_seq_len=max_seq_len, device=device)
    else:
        all_layers_bert_enc, pooling_output, \
        tokens, index_q, \
        q_tok_wp, q_lens, \
        cn_to_wp_index, wp_to_cn_index = get_bert_output(bert_model, bert_tokenizer,
                                                         q_tok_cn, max_seq_len=max_seq_len, device=device)

    if type(all_layers_bert_enc) != list:
        all_layers_bert_enc = [all_layers_bert_enc]

    # extract the embeddings
    q_emb = get_question_emb(all_layers_bert_enc, index_q, q_lens,
                             bert_config.hidden_size,
                             n_layers_bert_out_q, device)

    if h:
        h_emb = get_header_emb(all_layers_bert_enc, index_h, h_lens, h_nums,
                               bert_config.hidden_size,
                               n_layers_bert_out_h, device)
        return q_emb, q_lens, \
               h_emb, h_lens, h_nums, \
               q_tok_wp, cn_to_wp_index, wp_to_cn_index
    else:
        return q_emb, q_lens, \
               q_tok_wp, cn_to_wp_index, wp_to_cn_index

def get_char_emb(model, tokenizer, q_ch, h_ch, device=-1):
    q_idx = []
    q_lens = []
    max_q_len = 0
    for q in q_ch:
        q_idx.append(tokenizer.convert_to_index(q))
        q_lens.append(len(q))
        max_q_len = max(max_q_len, len(q))
    # padding
    for b in range(len(q_ch)):
        while len(q_idx[b]) < max_q_len:
            q_idx[b].append(tokenizer.lookup(tokenizer.pad_token))
    q_idx = torch.tensor(q_idx, dtype=torch.long)

    h_idx = []
    h_lens = []
    max_h_len = 0
    for hs in h_ch:
        for h in hs:
            h_idx.append(tokenizer.convert_to_index(h))
            h_lens.append(len(h))
            max_h_len = max(max_h_len, len(h))
    # padding
    for b in range(len(h_idx)):
        while len(h_idx[b]) < max_h_len:
            h_idx[b].append(tokenizer.lookup(tokenizer.pad_token))
    h_idx = torch.tensor(h_idx, dtype=torch.long)

    if device !=-1:
        q_idx = q_idx.to(device)
        h_idx = h_idx.to(device)

    q_emb = model.embed_ch(q_idx)
    h_emb = model.embed_ch(h_idx)
    return q_emb, q_lens, h_emb, h_lens

def encode_question(encoder, q_emb, lens, init_states=None):
    """
    :param encoder: lstm
    :param q_emb: [bs, max_q_len, d_emb]
    :param lens: [bs]
    :return: [bs, max_q_len, d_h]
    """
    q_enc, _ = encoder(q_emb, lens, init_states=init_states)
    return q_enc

def encode_header(encoder, h_emb, h_lens, h_nums, pooling_type="last", init_states=None):
    """
    :param encoder: lstm
    :param h_emb:  [bs, max_h_len, d_emb]
    :param h_lens: [total_h_num]
    :param h_nums: [bs]
    :param pooling_type: ["avg", "max", "last"]
    :return: [bs, max_h_num, d_h]
    """
    h_enc, _ = encoder(h_emb, h_lens, init_states=init_states)
    h_pooling = pooling(h_enc, h_lens, pooling_type)

    # Re-pack according to the batch
    bs = len(h_nums)
    max_h_num = max(h_nums)
    d_h = h_pooling.size(-1)

    packed_h_pooling =  torch.zeros(bs, max_h_num, d_h)
    if h_pooling.is_cuda:
        packed_h_pooling = packed_h_pooling.to(h_pooling.device)

    st_index = 0
    for i, h_num in enumerate(h_nums):
        packed_h_pooling[i, :h_num] = h_pooling[st_index : (st_index + h_num)]
        st_index += h_num
    return packed_h_pooling

def pooling(emb, lens, type):
    assert type in ["last", "avg", "max"]

    if type == "last":
        bs = len(emb)
        d_h = emb.size(-1)
        pooling_emb = torch.zeros(bs, d_h)

        if emb.is_cuda:
            pooling_emb = pooling_emb.to(emb.device)
        for i in range(bs):
            pooling_emb[i] = emb[i, lens[i]-1]
    else:
        mask = build_mask(emb, lens)
        emb = emb.masked_fill(mask==0, -float("inf"))
        pooling_emb = emb.max(dim=1)[0] if type == "max" else emb.mean(dim=1)[0]

    return pooling_emb

def build_mask(seq, seq_lens, dim=-2):
    mask = torch.zeros_like(seq)
    if dim == -1:
        mask.transpose_(-2, -1)
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    if dim == -1:
        mask.transpose_(-2, -1)
    return mask

def loss_sel_num(score_sel_num, gold_sel_num):
    f = nn.CrossEntropyLoss()
    gold_sel_num = torch.tensor(gold_sel_num)
    if score_sel_num.is_cuda:
        gold_sel_num = gold_sel_num.to(score_sel_num.device)
    return f(score_sel_num, gold_sel_num)

def loss_sel_col(score_sel_col, gold_sel_col):
    f = nn.BCELoss()
    index_mat = torch.zeros_like(score_sel_col)
    for b, cols in enumerate(gold_sel_col):
        for col in cols:
            index_mat[b, col] = 1.0
    prob = torch.sigmoid(score_sel_col)
    return f(prob, index_mat)

def loss_sel_agg(score_sel_agg, gold_sel_agg, gold_sel_num):
    f = nn.CrossEntropyLoss()
    loss = 0
    for b, col_num in enumerate(gold_sel_num):
        if col_num > 0:
            gold = torch.tensor(gold_sel_agg[b])
            if score_sel_agg.is_cuda:
                gold = gold.to(score_sel_agg.device)
            loss += f(score_sel_agg[b][: col_num], gold)
    return loss

def loss_where_join(score_where_join, gold_where_join):
    f = nn.CrossEntropyLoss()
    gold_where_join = torch.tensor(gold_where_join)
    if score_where_join.is_cuda:
        gold_where_join = gold_where_join.to(score_where_join.device)
    return f(score_where_join, gold_where_join)

def loss_where_num(score_where_num, gold_where_num):
    f = nn.CrossEntropyLoss()
    gold_where_num = torch.tensor(gold_where_num)
    if score_where_num.is_cuda:
        gold_where_num = gold_where_num.to(score_where_num.device)
    return f(score_where_num, gold_where_num)

def loss_where_col(score_where_col, gold_where_col):
    f = nn.BCELoss()
    index_mat = torch.zeros_like(score_where_col)
    for b, cols in enumerate(gold_where_col):
        for col in cols:
            index_mat[b, col] = 1.0
    prob = torch.sigmoid(score_where_col)
    return f(prob, index_mat)

def loss_where_agg(score_where_agg, gold_where_agg, gold_where_num):
    f = nn.CrossEntropyLoss()
    loss = 0
    for b, col_num in enumerate(gold_where_num):
        if col_num > 0:
            gold = torch.tensor(gold_where_agg[b])
            if score_where_agg.is_cuda:
                gold = gold.to(score_where_agg.device)
            loss += f(score_where_agg[b][: col_num], gold)
    return loss

def loss_where_op(score_where_op, gold_where_op, gold_where_num):
    f = nn.CrossEntropyLoss()
    loss = 0
    for b, col_num in enumerate(gold_where_num):
        if col_num > 0:
            gold = torch.tensor(gold_where_op[b])
            if score_where_op.is_cuda:
                gold = gold.to(score_where_op.device)
            loss += f(score_where_op[b][: col_num], gold)
    return loss

def loss_where_val(score_where_val, gold_where_val_index, gold_where_num):
    """
    :param score_where_val: [bs, max_where_num, max_q_len, 2]
    :param gold_where_val_index: [[[st_11, ed_11], [st_12, ed_12], ...], ... [[st_b1, ed_b1], [st_b2, ed_b2], ...]]
    """
    f = nn.CrossEntropyLoss()
    loss = 0
    for b, gold in enumerate(gold_where_val_index):
        col_num = gold_where_num[b]
        if col_num == 0:
            continue
        gold = torch.tensor(gold)
        if score_where_val.is_cuda:
            gold = gold.to(score_where_val.device)
        gold_st = gold[:, 0]
        gold_ed = gold[:, 1]

        loss += f(score_where_val[b, :col_num, :, 0], gold_st)
        loss += f(score_where_val[b, :col_num, :, 1], gold_ed)
    return loss

def loss_ord(score_ord, gold_ord):
    f = nn.CrossEntropyLoss()
    gold_ord = torch.tensor(gold_ord)
    if score_ord.is_cuda:
        gold_ord = gold_ord.to(score_ord.device)
    return f(score_ord, gold_ord)

def loss_ord_col(score_ord_col, gold_ord_col, gold_ord):
    f = nn.CrossEntropyLoss()
    loss = 0
    for i in range(score_ord_col.shape[0]):
        if gold_ord[i] == 2:
            continue
        _score_ord_col = score_ord_col[i].unsqueeze(0)
        _gold_ord_col = gold_ord_col[i]
        _gold_ord_col = torch.tensor(_gold_ord_col).unsqueeze(0)
        if _score_ord_col.is_cuda:
            _gold_ord_col = _gold_ord_col.to(_score_ord_col.device)
        loss += f(_score_ord_col, _gold_ord_col)
    return loss

def loss_ord_limit(score_ord_limit, gold_ord_limit, gold_ord):
    f = nn.CrossEntropyLoss()
    loss = 0
    for i in range(score_ord_limit.shape[0]):
        if gold_ord[i] == 2:
            continue
        _score_ord_limit = score_ord_limit[i].unsqueeze(0)
        _gold_ord_limit = gold_ord_limit[i]
        _gold_ord_limit = torch.tensor(_gold_ord_limit).unsqueeze(0)
        if _score_ord_limit.is_cuda:
            _gold_ord_limit = _gold_ord_limit.to(_score_ord_limit.device)
        loss += f(_score_ord_limit, _gold_ord_limit)
    return loss

def loss_wikisql(score_sel_col, score_sel_agg, score_where_num, score_where_col, score_where_op, score_where_val,
                 gold_sel_col, gold_sel_agg, gold_where_num, gold_where_col, gold_where_op, gold_where_val_index):
    loss = 0
    # In wikisql, each sample selects only one column.

    loss += loss_where_num(score_sel_col, gold_sel_col)
    loss += loss_where_num(score_sel_agg, gold_sel_agg)
    loss += loss_where_num(score_where_num, gold_where_num)
    loss += loss_where_col(score_where_col, gold_where_col)
    loss += loss_where_op(score_where_op, gold_where_op, gold_where_num)
    loss += loss_where_val(score_where_val, gold_where_val_index, gold_where_num)
    return loss

def loss_esql(score_sel_num, score_sel_col, score_sel_agg, score_where_join, score_where_num, score_where_col,
                score_where_agg, score_where_op, score_where_val, score_ord, score_ord_col, score_ord_limit,
                gold_sel_num, gold_sel_col, gold_sel_agg, gold_where_join, gold_where_num, gold_where_col,
                gold_where_agg, gold_where_op, gold_where_val_index, gold_ord, gold_ord_col, gold_ord_limit):
    loss = 0
    loss += loss_sel_num(score_sel_num, gold_sel_num)
    loss += loss_sel_col(score_sel_col, gold_sel_col)
    loss += loss_sel_agg(score_sel_agg, gold_sel_agg, gold_sel_num)
    loss += loss_where_join(score_where_join, gold_where_join)
    loss += loss_where_num(score_where_num, gold_where_num)
    loss += loss_where_col(score_where_col, gold_where_col)
    loss += loss_where_agg(score_where_agg, gold_where_agg, gold_where_num)
    loss += loss_where_op(score_where_op, gold_where_op, gold_where_num)
    loss += loss_where_val(score_where_val, gold_where_val_index, gold_where_num)
    loss += loss_ord(score_ord, gold_ord)
    loss += loss_ord_col(score_ord_col, gold_ord_col, gold_ord)
    loss += loss_ord_limit(score_ord_limit, gold_ord_limit, gold_ord)
    return loss