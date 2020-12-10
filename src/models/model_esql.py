# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/3
# @Author  : Yongrui Chen
# @File    : model_esql.py
# @Software: PyCharm
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import argmax, array
sys.path.append("..")
from src.models.modules import SelectNumber, SelectColumn, SelectMultipleAggregation
from src.models.modules import WhereJoiner, WhereNumber, WhereColumn, WhereOperator, WhereValue, WhereAggregation
from src.models.modules import OrderByColumn, OrderByLimit, OrderByOrder
from src.utils.utils import pred_sel_num, pred_sel_col, pred_where_num, pred_where_op, pred_where_col


class Seq2SQL(nn.Module):
    def __init__(self, d_in, d_in_ch, d_h, d_f, n_layers, dropout_prob, ch_vocab_size,
                 n_op, n_agg, n_limit, max_sel_num, max_where_num, pooling_type="last"):
        super(Seq2SQL, self).__init__()
        self.d_in = d_in
        self.d_in_ch = d_in_ch
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.max_sel_num = max_sel_num
        self.max_where_num = max_where_num
        self.n_op = n_op
        self.n_agg = n_agg
        self.n_limit = n_limit

        self.embed_ch = nn.Embedding(ch_vocab_size, d_in_ch)

        # SELECT
        self.m_sel_num = SelectNumber(d_in, d_h, n_layers, dropout_prob, pooling_type, max_sel_num)
        self.m_sel_col = SelectColumn(d_in, d_h, n_layers, dropout_prob, pooling_type)
        self.m_sel_agg = SelectMultipleAggregation(d_in, d_h, n_layers, dropout_prob, n_agg, pooling_type, max_sel_num)

        # WHERE
        self.m_where_join = WhereJoiner(d_in, d_h, n_layers, dropout_prob)
        self.m_where_num = WhereNumber(d_in, d_in_ch, d_h, n_layers, dropout_prob, pooling_type, max_where_num)
        self.m_where_col = WhereColumn(d_in, d_in_ch, d_h, n_layers, dropout_prob, pooling_type)
        self.m_where_agg = WhereAggregation(d_in, d_h, n_layers, dropout_prob, n_agg, pooling_type, max_where_num)
        self.m_where_op = WhereOperator(d_in, d_h, n_layers, dropout_prob, n_op, pooling_type, max_where_num)
        self.m_where_val = WhereValue(d_in, d_in_ch, d_h, d_f, n_layers, dropout_prob, n_op, pooling_type, max_where_num)

        # Orderby
        self.m_ord = OrderByOrder(d_in, d_h, n_layers, dropout_prob)
        self.m_ord_col = OrderByColumn(d_in, d_h, n_layers, dropout_prob, pooling_type)
        self.m_ord_limit = OrderByLimit(d_in, d_h, n_layers, dropout_prob, n_limit)


    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch, q_feature,
                gold_sel_num=None, gold_sel_col=None, gold_where_num=None,
                gold_where_col=None, gold_where_op=None):
        bs = len(q_emb)

        # select number
        score_sel_num = self.m_sel_num(q_emb, q_lens, h_emb, h_lens, h_nums)
        if gold_sel_num:
            now_sel_num = gold_sel_num
        else:
            now_sel_num = pred_sel_num(score_sel_num)

        # select columns
        score_sel_col = self.m_sel_col(q_emb, q_lens, h_emb, h_lens, h_nums)

        if gold_sel_col:
            now_sel_col = gold_sel_col
        else:
            now_sel_col = pred_sel_col(now_sel_num, score_sel_col)

        # select aggregation
        score_sel_agg = self.m_sel_agg(q_emb, q_lens, h_emb, h_lens, h_nums, now_sel_col)

        # where joiner
        score_where_join = self.m_where_join(q_emb, q_lens)

        # where number
        score_where_num = self.m_where_num(q_emb, q_lens, h_emb, h_lens, h_nums,
                                           q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch)

        if gold_where_num:
            now_where_num = gold_where_num
        else:
            now_where_num = pred_where_num(score_where_num)

        # where column
        score_where_col = self.m_where_col(q_emb, q_lens, h_emb, h_lens, h_nums,
                                           q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch)

        if gold_where_col:
            now_where_col = gold_where_col
        else:
            now_where_col = pred_where_col(now_where_num, score_where_col)

        # where aggregation
        score_where_agg = self.m_where_agg(q_emb, q_lens, h_emb, h_lens, h_nums, now_where_col)

        # where operator
        score_where_op = self.m_where_op(q_emb, q_lens, h_emb, h_lens, h_nums, now_where_col)

        if gold_where_op:
            now_where_op = gold_where_op
        else:
            now_where_op = pred_where_op(now_where_num, score_where_op)

        # where value
        score_where_val = self.m_where_val(q_emb, q_lens, h_emb, h_lens, h_nums,
                                           q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                           now_where_col, now_where_op,
                                           q_feature)

        # order by
        score_ord = self.m_ord(q_emb, q_lens)

        score_ord_col = self.m_ord_col(q_emb, q_lens, h_emb, h_lens, h_nums)

        score_ord_limit = self.m_ord_limit(q_emb, q_lens)

        return score_sel_num, score_sel_col, score_sel_agg, \
               score_where_join, score_where_num, \
               score_where_col, score_where_agg, score_where_op, score_where_val, \
               score_ord, score_ord_col, score_ord_limit