# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : modules.py
# @Software: PyCharm
"""
import sys
import torch
import torch.nn as nn
sys.path.append("..")
from src.models.nn_utils import encode_question, encode_header, build_mask


class LSTM(nn.Module):

    def __init__(self, d_input, d_h, n_layers=1, batch_first=True, birnn=True, dropout=0.3):
        super(LSTM, self).__init__()

        n_dir = 2 if birnn else 1
        self.init_h = nn.Parameter(torch.Tensor(n_layers * n_dir, d_h))
        self.init_c = nn.Parameter(torch.Tensor(n_layers * n_dir, d_h))

        INI = 1e-2
        torch.nn.init.uniform_(self.init_h, -INI, INI)
        torch.nn.init.uniform_(self.init_c, -INI, INI)

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_h,
            num_layers=n_layers,
            bidirectional=birnn,
            batch_first=not batch_first
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, seq_lens=None, init_states=None):

        bs = seqs.size(0)
        bf = self.lstm.batch_first

        if not bf:
            seqs = seqs.transpose(0, 1)

        seqs = self.dropout(seqs)


        size = (self.init_h.size(0), bs, self.init_h.size(1))
        if init_states is None:
            init_states = (self.init_h.unsqueeze(1).expand(*size).contiguous(),
                           self.init_c.unsqueeze(1).expand(*size).contiguous())

        if seq_lens is not None:
            assert bs == len(seq_lens)
            sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
            seq_lens = [seq_lens[i] for i in sort_ind]
            seqs = self.reorder_sequence(seqs, sort_ind, bf)
            init_states = self.reorder_init_states(init_states, sort_ind)

            packed_seq = nn.utils.rnn.pack_padded_sequence(seqs, seq_lens)
            packed_out, final_states = self.lstm(packed_seq, init_states)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

            back_map = {ind: i for i, ind in enumerate(sort_ind)}
            reorder_ind = [back_map[i] for i in range(len(seq_lens))]
            lstm_out = self.reorder_sequence(lstm_out, reorder_ind, bf)
            final_states = self.reorder_init_states(final_states, reorder_ind)
        else:
            lstm_out, final_states = self.lstm(seqs)
        return lstm_out.transpose(0, 1), final_states

    def reorder_sequence(self, seqs, order, batch_first=False):
        """
        seqs: [T, B, D] if not batch_first
        order: list of sequence length
        """
        batch_dim = 0 if batch_first else 1
        assert len(order) == seqs.size()[batch_dim]
        order = torch.LongTensor(order).to(seqs.device)
        sorted_seqs = seqs.index_select(index=order, dim=batch_dim)
        return sorted_seqs

    def reorder_init_states(self, states, order):
        """
        lstm_states: (H, C) of tensor [layer, batch, hidden]
        order: list of sequence length
        """
        assert isinstance(states, tuple)
        assert len(states) == 2
        assert states[0].size() == states[1].size()
        assert len(order) == states[0].size()[1]

        order = torch.LongTensor(order).to(states[0].device)
        sorted_states = (states[0].index_select(index=order, dim=1),
                         states[1].index_select(index=order, dim=1))
        return sorted_states

class SelectNumber(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, pooling_type, max_select_num):
        super(SelectNumber, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.pooling_type = pooling_type
        self.max_select_num = max_select_num

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att_q = nn.Linear(d_h, 1)
        self.W_att_h = nn.Linear(d_h, 1)
        self.W_hidden = nn.Linear(d_h, d_h * n_layers)
        self.W_cell = nn.Linear(d_h, d_h * n_layers)
        self.W_out = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.Tanh(),
            nn.Linear(d_h, self.max_select_num + 1)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums):

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        bs = len(q_lens)

        # self-attention for header
        # [bs, max_h_num]
        att_weights_h = self.W_att_h(h_pooling).squeeze(2)
        att_mask_h = build_mask(att_weights_h, q_lens, dim=-2)
        att_weights_h = self.softmax(att_weights_h.masked_fill(att_mask_h == 0, -float("inf")))

        # [bs, d_h]
        h_context = torch.mul(h_pooling, att_weights_h.unsqueeze(2)).sum(1)

        # [bs, d_h] -> [bs, 2 * d_h]
        # enlarge because there are two layers.
        hidden = self.W_hidden(h_context)
        hidden = hidden.view(bs, self.n_layers * 2, int(self.d_h / 2))
        hidden = hidden.transpose(0, 1).contiguous()

        cell = self.W_cell(h_context)
        cell = cell.view(bs, self.n_layers * 2, int(self.d_h / 2))
        cell = cell.transpose(0, 1).contiguous()

        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens, init_states=(hidden, cell))

        # self-attention for question
        # [bs, max_q_len]
        att_weights_q = self.W_att_q(q_enc).squeeze(2)
        att_mask_q = build_mask(att_weights_q, q_lens, dim=-2)
        att_weights_q = self.softmax(att_weights_q.masked_fill(att_mask_q == 0, -float("inf")))

        q_context = torch.mul(q_enc, att_weights_q.unsqueeze(2).expand_as(q_enc)).sum(dim=1)

        # [bs, max_select_num + 1]
        score_sel_num = self.W_out(q_context)
        return score_sel_num


class SelectColumn(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, pooling_type):
        super(SelectColumn, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.pooling_type = pooling_type

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_q = nn.Linear(d_h, d_h)
        self.W_h = nn.Linear(d_h, d_h)
        self.W_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * d_h, 1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        # [bs, max_h_num, max_q_len]
        att_weights = torch.bmm(h_pooling, self.W_att(q_enc).transpose(1, 2))
        att_mask = build_mask(att_weights, q_lens, dim=-1)
        att_weights = self.softmax(att_weights.masked_fill(att_mask==0, -float("inf")))

        # att_weights: -> [bs, max_h_num, max_q_len, 1]
        # q_enc: -> [bs, 1, max_q_len, d_h]
        # [bs, max_h_num, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)

        # [bs, max_h_num, d_h * 2]
        comb_context = torch.cat([self.W_q(q_context), self.W_h(h_pooling)], dim=-1)

        # [bs, max_h_num]
        score_sel_col = self.W_out(comb_context).squeeze(2)

        # mask
        for b, h_num in enumerate(h_nums):
            score_sel_col[b, h_num:] = -float("inf")
        return score_sel_col


class SelectAggregation(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, n_agg, pooling_type, max_sel_num):
        super(SelectAggregation, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.n_agg = n_agg
        self.pooling_type = pooling_type
        self.max_sel_num = max_sel_num

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_out = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.Tanh(),
            nn.Linear(d_h, self.n_agg)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums, sel_col):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        bs = len(q_emb)
        h_pooling_sel = h_pooling[list(range(bs)), sel_col]

        att_weights = torch.bmm(self.W_att(q_enc), h_pooling_sel.unsqueeze(2)).squeeze(2)
        att_mask = build_mask(att_weights, q_lens, dim=-2)
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # att_weights: [bs, max_sel_num, max_q_len] -> [bs, max_sel_num, max_q_len, 1]
        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        q_context = torch.mul(q_enc, att_weights.unsqueeze(2).expand_as(q_enc)).sum(dim=1)

        # [bs, max_sel_num, n_agg]
        score_sel_agg = self.W_out(q_context)
        return score_sel_agg


class SelectMultipleAggregation(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, n_agg, pooling_type, max_sel_num):
        super(SelectMultipleAggregation, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.n_agg = n_agg
        self.pooling_type = pooling_type
        self.max_sel_num = max_sel_num

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_q = nn.Linear(d_h, d_h)
        self.W_h = nn.Linear(d_h, d_h)
        self.W_out = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.Tanh(),
            nn.Linear(d_h, self.n_agg)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums, sel_cols):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        padding_t = torch.zeros_like(h_pooling[0][0]).unsqueeze(0)

        h_pooling_sel = []
        for b, cols in enumerate(sel_cols):
            if len(cols) > 0:
                h_tmp = [h_pooling[b][cols, :]]
            else:
                h_tmp = []
            h_tmp += [padding_t] * (self.max_sel_num - len(cols))
            h_tmp = torch.cat(h_tmp, dim=0)
            h_pooling_sel.append(h_tmp)
        # [bs, max_sel_num, d_h]
        h_pooling_sel = torch.stack(h_pooling_sel)

        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        # h_pooling_sel: [bs, max_sel_num, d_h] -> [bs, max_sel_num, d_h, 1]
        # [bs, max_sel_num, max_q_len]
        att_weights = torch.matmul(
            self.W_att(q_enc).unsqueeze(1),
            h_pooling_sel.unsqueeze(3)
        ).squeeze(3)
        att_mask = build_mask(att_weights, q_lens, dim=-1)
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # att_weights: [bs, max_sel_num, max_q_len] -> [bs, max_sel_num, max_q_len, 1]
        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)

        # [bs, max_sel_num, n_agg]
        score_sel_agg = self.W_out(torch.cat([self.W_q(q_context), self.W_h(h_pooling_sel)], dim=2))
        return score_sel_agg


class WhereJoiner(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob):
        super(WhereJoiner, self).__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, 1)
        self.W_out = nn.Sequential(nn.Linear(d_h, d_h),
                                   nn.Tanh(),
                                   nn.Linear(d_h, 3))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, q_emb, q_lens):

        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        #  self-atttention for question
        #  [bs, max_q_len]
        att_weights_q = self.W_att(q_enc).squeeze(2)
        att_mask_q = build_mask(att_weights_q, q_lens, dim=-2)
        att_weights_q = att_weights_q.masked_fill(att_mask_q == 0, -float('inf'))
        att_weights_q = self.softmax(att_weights_q)

        #  [bs, d_h]
        q_context = torch.mul(
            q_enc,
            att_weights_q.unsqueeze(2).expand_as(q_enc)
        ).sum(dim=1)
        where_op_logits = self.W_out(q_context)
        return where_op_logits


class WhereNumber(nn.Module):
    def __init__(self, d_in, d_in_ch, d_h, n_layers, dropout_prob, pooling_type, max_where_num):
        super(WhereNumber, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.pooling_type = pooling_type
        self.max_where_num = max_where_num

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.q_encoder_ch = LSTM(d_input=d_in_ch, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder_ch = LSTM(d_input=d_in_ch, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att_q = nn.Linear(d_h, 1)
        self.W_att_h = nn.Linear(d_h, 1)
        self.W_hidden = nn.Linear(d_h, d_h * n_layers)
        self.W_cell = nn.Linear(d_h, d_h * n_layers)

        self.W_att_q_ch = nn.Linear(d_h, 1)
        self.W_att_h_ch = nn.Linear(d_h, 1)
        self.W_hidden_ch = nn.Linear(d_h, d_h * n_layers)
        self.W_cell_ch = nn.Linear(d_h, d_h * n_layers)

        self.W_out = nn.Sequential(
            nn.Linear(d_h * 2, d_h),
            nn.Tanh(),
            nn.Linear(d_h, self.max_where_num + 1)
        )

        self.softmax = nn.Softmax(dim=-1)

    def get_context(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                    q_encoder, h_encoder, W_att_q, W_att_h, W_hidden, W_cell):
        # [bs, max_h_num, d_h]
        h_pooling = encode_header(h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        bs = len(q_lens)

        # self-attention for header
        # [bs, max_h_num]
        att_weights_h = W_att_h(h_pooling).squeeze(2)
        att_mask_h = build_mask(att_weights_h, q_lens, dim=-2)
        att_weights_h = self.softmax(att_weights_h.masked_fill(att_mask_h == 0, -float("inf")))

        # [bs, d_h]
        h_context = torch.mul(h_pooling, att_weights_h.unsqueeze(2)).sum(1)

        # [bs, d_h] -> [bs, 2 * d_h]
        # enlarge because there are two layers.
        hidden = W_hidden(h_context)
        hidden = hidden.view(bs, self.n_layers * 2, int(self.d_h / 2))
        hidden = hidden.transpose(0, 1).contiguous()

        cell = W_cell(h_context)
        cell = cell.view(bs, self.n_layers * 2, int(self.d_h / 2))
        cell = cell.transpose(0, 1).contiguous()

        # [bs, max_q_len, d_h]
        q_enc = encode_question(q_encoder, q_emb, q_lens, init_states=(hidden, cell))

        # self-attention for question
        # [bs, max_q_len]
        att_weights_q = W_att_q(q_enc).squeeze(2)
        att_mask_q = build_mask(att_weights_q, q_lens, dim=-2)
        att_weights_q = self.softmax(att_weights_q.masked_fill(att_mask_q == 0, -float("inf")))

        q_context = torch.mul(q_enc, att_weights_q.unsqueeze(2).expand_as(q_enc)).sum(dim=1)
        return q_context

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                    q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch):

        q_context = self.get_context(q_emb, q_lens, h_emb, h_lens, h_nums,
                                     self.q_encoder, self.h_encoder, self.W_att_q,
                                     self.W_att_h, self.W_hidden, self.W_cell)
        q_context_ch = self.get_context(q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch, h_nums,
                                     self.q_encoder_ch, self.h_encoder_ch, self.W_att_q_ch,
                                     self.W_att_h_ch, self.W_hidden_ch, self.W_cell_ch)
        # [bs, max_where_num + 1]
        score_sel_num = self.W_out(torch.cat([q_context, q_context_ch], dim=-1))
        return score_sel_num


class WhereColumn(nn.Module):
    def __init__(self, d_in, d_in_ch, d_h, n_layers, dropout_prob, pooling_type):
        super(WhereColumn, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.pooling_type = pooling_type

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_q = nn.Linear(d_h, d_h)
        self.W_h = nn.Linear(d_h, d_h)

        self.q_encoder_ch = LSTM(d_input=d_in_ch, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder_ch = LSTM(d_input=d_in_ch, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att_ch = nn.Linear(d_h, d_h)
        self.W_q_ch = nn.Linear(d_h, d_h)
        self.W_h_ch = nn.Linear(d_h, d_h)

        self.W_out = nn.Sequential(nn.Tanh(), nn.Linear(4 * d_h, 1))

        self.softmax = nn.Softmax(dim=-1)

    def get_context(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                    q_encoder, h_encoder, W_att, W_q, W_h):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(q_encoder, q_emb, q_lens)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        # [bs, max_h_num, max_q_len]
        att_weights = torch.bmm(h_pooling, W_att(q_enc).transpose(1, 2))
        att_mask = build_mask(att_weights, q_lens, dim=-1)
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # att_weights: -> [bs, max_h_num, max_q_len, 1]
        # q_enc: -> [bs, 1, max_q_len, d_h]
        # [bs, max_h_num, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)

        return W_q(q_context), W_h(h_pooling)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch):
        q_context, h_pooling = self.get_context(q_emb, q_lens, h_emb, h_lens, h_nums,
                                                self.q_encoder, self.h_encoder, self.W_att,
                                                self.W_q, self.W_h)

        q_context_ch, h_pooling_ch = self.get_context(q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch, h_nums,
                                                    self.q_encoder_ch, self.h_encoder_ch, self.W_att_ch,
                                                    self.W_q_ch, self.W_h_ch)

        # [bs, max_h_num, d_h * 2]
        comb_context = torch.cat([q_context, q_context_ch, h_pooling, h_pooling_ch], dim=-1)

        # [bs, max_h_num]
        score_where_col = self.W_out(comb_context).squeeze(2)
        for b, h_num in enumerate(h_nums):
            score_where_col[b, h_num:] = -float("inf")
        return score_where_col


class WhereOperator(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, n_op, pooling_type, max_where_num):
        super(WhereOperator, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.n_op = n_op
        self.pooling_type = pooling_type
        self.max_where_num = max_where_num

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_q = nn.Linear(d_h, d_h)
        self.W_h = nn.Linear(d_h, d_h)
        self.W_out = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.Tanh(),
            nn.Linear(d_h, self.n_op)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums, where_cols):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        padding_t = torch.zeros_like(h_pooling[0][0]).unsqueeze(0)

        h_pooling_where = []
        for b, cols in enumerate(where_cols):
            if len(cols) > 0:
                h_tmp = [h_pooling[b][cols, :]]
            else:
                h_tmp = []
            h_tmp += [padding_t] * (self.max_where_num - len(cols))
            h_tmp = torch.cat(h_tmp, dim=0)
            h_pooling_where.append(h_tmp)
        # [bs, max_where_num, d_h]
        h_pooling_where = torch.stack(h_pooling_where)

        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        # h_pooling_where: [bs, max_where_num, d_h] -> [bs, max_where_num, d_h, 1]
        # [bs, max_where_num, max_q_len]
        att_weights = torch.matmul(
            self.W_att(q_enc).unsqueeze(1),
            h_pooling_where.unsqueeze(3)
        ).squeeze(3)
        att_mask = build_mask(att_weights, q_lens, dim=-1)
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # att_weights: [bs, max_where_num, max_q_len] -> [bs, max_where_num, max_q_len, 1]
        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)

        # [bs, max_where_num, n_agg]
        score_where_op = self.W_out(torch.cat([self.W_q(q_context), self.W_h(h_pooling_where)], dim=2))
        return score_where_op


class WhereAggregation(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, n_agg, pooling_type, max_where_num):
        super(WhereAggregation, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.n_agg = n_agg
        self.pooling_type = pooling_type
        self.max_where_num = max_where_num

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h/2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_q = nn.Linear(d_h, d_h)
        self.W_h = nn.Linear(d_h, d_h)
        self.W_out = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.Tanh(),
            nn.Linear(d_h, self.n_agg)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums, where_cols):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        padding_t = torch.zeros_like(h_pooling[0][0]).unsqueeze(0)

        h_pooling_where = []
        for b, cols in enumerate(where_cols):
            if len(cols) > 0:
                h_tmp = [h_pooling[b][cols, :]]
            else:
                h_tmp = []
            h_tmp += [padding_t] * (self.max_where_num - len(cols))
            h_tmp = torch.cat(h_tmp, dim=0)
            h_pooling_where.append(h_tmp)
        # [bs, max_where_num, d_h]
        h_pooling_where = torch.stack(h_pooling_where)

        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        # h_pooling_where: [bs, max_where_num, d_h] -> [bs, max_where_num, d_h, 1]
        # [bs, max_where_num, max_q_len]
        att_weights = torch.matmul(
            self.W_att(q_enc).unsqueeze(1),
            h_pooling_where.unsqueeze(3)
        ).squeeze(3)
        att_mask = build_mask(att_weights, q_lens, dim=-1)
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # att_weights: [bs, max_where_num, max_q_len] -> [bs, max_where_num, max_q_len, 1]
        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)

        # [bs, max_where_num, n_agg]
        score_where_agg = self.W_out(torch.cat([self.W_q(q_context), self.W_h(h_pooling_where)], dim=2))
        return score_where_agg


class WhereValue(nn.Module):
    def __init__(self, d_in, d_in_ch, d_h, d_f, n_layers, dropout_prob, n_op, pooling_type, max_where_num):
        super(WhereValue, self).__init__()

        self.d_in = d_in
        self.d_h = d_h
        self.d_in_ch = d_in_ch
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.n_op = n_op
        self.pooling_type = pooling_type
        self.max_where_num = max_where_num

        self.d_f = d_f
        self.q_feature_embed = nn.Embedding(2, self.d_f)

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h + self.d_f, d_h)
        self.W_q = nn.Linear(d_h + self.d_f, d_h)
        self.W_h = nn.Linear(d_h, d_h)

        self.q_encoder_ch = LSTM(d_input=d_in_ch, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)
        self.h_encoder_ch = LSTM(d_input=d_in_ch, d_h=int(d_h / 2), n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att_ch = nn.Linear(d_h, d_h)
        self.W_q_ch = nn.Linear(d_h, d_h)
        self.W_h_ch = nn.Linear(d_h, d_h)

        self.W_op = nn.Linear(n_op, d_h)

        self.W_out = nn.Sequential(
            nn.Linear(6 * d_h + self.d_f, d_h),
            nn.Tanh(),
            nn.Linear(d_h, 2)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch, where_cols, where_ops,
                q_feature):
        bs = len(q_emb)
        max_q_len = max(q_lens)

        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)
        for b, f in enumerate(q_feature):
            while len(f) < max_q_len:
                q_feature[b].append(0)
        q_feature = torch.tensor(q_feature)
        if q_enc.is_cuda:
            q_feature = q_feature.to(q_enc.device)

        q_feature_enc = self.q_feature_embed(q_feature)

        q_enc = torch.cat([q_enc, q_feature_enc], -1)

        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder, h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        padding_t = torch.zeros_like(h_pooling[0][0]).unsqueeze(0)
        h_pooling_where = []
        for b, cols in enumerate(where_cols):
            if len(cols) > 0:
                h_tmp = [h_pooling[b][cols, :]]
            else:
                h_tmp = []
            h_tmp += [padding_t] * (self.max_where_num - len(cols))
            h_tmp = torch.cat(h_tmp, dim=0)
            h_pooling_where.append(h_tmp)
        # [bs, max_where_num, d_h]
        h_pooling_where = torch.stack(h_pooling_where)

        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        # h_pooling_where: [bs, max_where_num, d_h] -> [bs, max_where_num, d_h, 1]
        # [bs, max_where_num, max_q_len]
        att_weights = torch.matmul(
            self.W_att(q_enc).unsqueeze(1),
            h_pooling_where.unsqueeze(3)
        ).squeeze(3)
        att_mask = build_mask(att_weights, q_lens, dim=-1)
        att_weights = self.softmax(att_weights.masked_fill(att_mask == 0, -float("inf")))

        # att_weights: [bs, max_where_num, max_q_len] -> [bs, max_where_num, max_q_len, 1]
        # q_enc: [bs, max_q_len, d_h] -> [bs, 1, max_q_len, d_h]
        # [bs, max_where_num, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)

        q_enc_ch = encode_question(self.q_encoder_ch, q_emb_ch, q_lens_ch)

        # [bs, max_h_num, d_h]
        h_pooling_ch = encode_header(self.h_encoder_ch, h_emb_ch, h_lens_ch,
                                     h_nums, pooling_type=self.pooling_type)

        padding_t_ch = torch.zeros_like(h_pooling_ch[0][0]).unsqueeze(0)

        h_pooling_where_ch = []
        for b, cols in enumerate(where_cols):
            if len(cols) > 0:
                h_tmp = [h_pooling_ch[b][cols, :]]
            else:
                h_tmp = []
            h_tmp += [padding_t_ch] * (self.max_where_num - len(cols))
            h_tmp = torch.cat(h_tmp, dim=0)
            h_pooling_where_ch.append(h_tmp)
        h_pooling_where_ch = torch.stack(h_pooling_where_ch)

        att_weights_ch = torch.matmul(
            self.W_att_ch(q_enc_ch).unsqueeze(1),
            h_pooling_where_ch.unsqueeze(3)
        ).squeeze(3)
        att_mask_ch = build_mask(att_weights_ch, q_lens_ch, dim=-1)
        att_weights_ch = self.softmax(att_weights_ch.masked_fill(att_mask_ch == 0, -float("inf")))
        q_context_ch = torch.mul(att_weights_ch.unsqueeze(3), q_enc_ch.unsqueeze(1)).sum(dim=2)

        op_enc = []
        for b in range(bs):
            op_enc_tmp = torch.zeros(self.max_where_num, self.n_op)
            op = where_ops[b]
            idx_scatter = []
            op_len = len(op)
            for i in range(self.max_where_num):
                if i < op_len:
                    idx_scatter.append([op[i]])
                else:
                    idx_scatter.append([0])
            op_enc_tmp = op_enc_tmp.scatter(1, torch.tensor(idx_scatter), 1)
            op_enc.append(op_enc_tmp)
        op_enc = torch.stack(op_enc)
        if q_context.is_cuda:
            op_enc = op_enc.to(q_context.device)

        comb_context = torch.cat(
            [self.W_q(q_context),
             self.W_h(h_pooling_where),
             self.W_q_ch(q_context_ch),
             self.W_h_ch(h_pooling_where_ch),
             self.W_op(op_enc)],
            dim=2
        )
        comb_context = comb_context.unsqueeze(2).expand(-1, -1, q_enc.size(1), -1)
        q_enc = q_enc.unsqueeze(1).expand(-1, comb_context.size(1), -1, -1)

        # [bs, max_where_num, max_q_num, 2]
        score_where_val = self.W_out(torch.cat([comb_context, q_enc], dim=3))

        for b, l in enumerate(q_lens):
            if l < max_q_len:
                score_where_val[b, :, l:, :] = -float("inf")
        return score_where_val


class OrderByColumn(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, pooling_type):
        super(OrderByColumn, self).__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.pooling_type = pooling_type

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2),
                              n_layers=n_layers, batch_first=True
                              , dropout=dropout_prob, birnn=True)
        self.h_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2),
                              n_layers=n_layers, batch_first=True,
                              dropout=dropout_prob, birnn=True)

        self.W_att = nn.Linear(d_h, d_h)
        self.W_question = nn.Linear(d_h, d_h)
        self.W_header = nn.Linear(d_h, d_h)
        self.W_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * d_h, 1))
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums, mask=True):

        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)
        # [bs, max_h_num, d_h]
        h_pooling = encode_header(self.h_encoder,
                                  h_emb, h_lens, h_nums, pooling_type=self.pooling_type)

        # [bs, max_h_num, max_q_len]
        # torch.bmm: bs * ([max_header_len, d_h], [d_h, max_q_len])
        att_weights = torch.bmm(h_pooling, self.W_att(q_enc).transpose(1, 2))
        att_mask = build_mask(att_weights, h_nums, dim=-1)
        att_weights = att_weights.masked_fill(att_mask == 0, -float('inf'))
        att_weights = self.softmax(att_weights)

        # attention_weights: -> [bs, max_h_num, max_q_len, 1]
        # q_enc: -> [bs, 1, max_q_len, d_h]
        # [bs, max_h_num, d_h]
        q_context = torch.mul(att_weights.unsqueeze(3), q_enc.unsqueeze(1)).sum(dim=2)
        comb_context = torch.cat([self.W_question(q_context), self.W_header(h_pooling)], dim=-1)

        score_ord_col = self.W_out(comb_context).squeeze(2)

        if mask:
            for b, h_num in enumerate(h_nums):
                score_ord_col[b, h_num:] = -float('inf')
        return score_ord_col


class OrderByOrder(nn.Module):
    def __init__(self, d_in=300, d_h=100, n_layers=2, dropout_prob=0.3):
        super(OrderByOrder, self).__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.max_order = 3

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers,
                              batch_first=True, dropout=dropout_prob, birnn=True)

        self.W_att_question = nn.Linear(d_h, d_h)
        self.W_att = nn.Linear(d_h, 1)

        self.W_out = nn.Sequential(nn.Linear(d_h, d_h), nn.Tanh(),
                                   nn.Linear(d_h, self.max_order))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q_emb, q_lens):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_q_len, 1]
        att_weights = self.W_att(q_enc)
        # print(att_weights.shape)
        att_mask = build_mask(att_weights, q_lens, dim=-2)
        att_weights = att_weights.masked_fill(att_mask == 0, -float('inf'))
        # [bs, max_q_len, 1]
        att_weights = self.softmax(att_weights)
        # print(att_weights.shape)

        # [bs, d_h]
        q_context = torch.bmm(q_enc.transpose(1, 2), att_weights).squeeze(2)

        score_order = self.W_out(q_context)
        return score_order


class OrderByLimit(nn.Module):
    def __init__(self, d_in, d_h, n_layers, dropout_prob, n_limit):
        super(OrderByLimit, self).__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        # limit = 0 -> no limit        limit <- [1, 9]
        self.n_limit = n_limit

        self.q_encoder = LSTM(d_input=d_in, d_h=int(d_h / 2), n_layers=n_layers,
                              batch_first=True, dropout=dropout_prob, birnn=True)

        self.W_att_question = nn.Linear(d_h, d_h)
        self.W_att = nn.Linear(d_h, 1)

        self.W_out = nn.Sequential(nn.Linear(d_h, d_h), nn.Tanh(),
                                   nn.Linear(d_h, self.n_limit))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q_emb, q_lens):
        # [bs, max_q_len, d_h]
        q_enc = encode_question(self.q_encoder, q_emb, q_lens)

        # [bs, max_q_len, 1]
        att_weights = self.W_att(q_enc)
        att_mask = build_mask(att_weights, q_lens, dim=-2)
        att_weights = att_weights.masked_fill(att_mask == 0, -float('inf'))
        # [bs, max_q_len, 1]
        att_weights = self.softmax(att_weights)

        # [bs, d_h]
        q_context = torch.bmm(q_enc.transpose(1, 2), att_weights).squeeze(2)
        score_limit = self.W_out(q_context)
        return score_limit