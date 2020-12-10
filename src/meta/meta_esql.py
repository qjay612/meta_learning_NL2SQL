# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/3
# @Author  : Yongrui Chen
# @File    : meta_esql.py
# @Software: PyCharm
"""

import torch
import pickle
from torch import nn

from src.models.model_esql import Seq2SQL
from src.utils.utils import load_bert_from_tf
from src.utils.utils_esql import train_step

class Learner(nn.Module):
    def __init__(self, args):
        super(Learner, self).__init__()
        self.n_layers_bert_out = args.n_layers_bert_out
        self.max_seq_len = args.max_seq_len
        self.h_num_limit = args.h_num_limit
        self.bert_model, self.bert_tokenizer, self.bert_config = load_bert_from_tf(args.bert_path)

        self.ch_tokenizer = pickle.load(open(args.ch_vocab_path, "rb"))
        self.model = Seq2SQL(d_in=self.bert_config.hidden_size * args.n_layers_bert_out,
                             d_in_ch=args.d_in_ch,
                             d_h=args.d_h,
                             d_f=args.d_f,
                             n_layers=args.n_layers,
                             dropout_prob=args.dropout,
                             ch_vocab_size=len(self.ch_tokenizer),
                             n_op=args.n_op,
                             n_agg=args.n_agg,
                             n_limit=args.n_limit,
                             max_sel_num=args.max_sel_num,
                             max_where_num=args.max_where_num)


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.args = args
        self.update_lr = args.update_lr
        self.update_bert_lr = args.update_bert_lr
        self.meta_lr = args.meta_lr
        self.meta_bert_lr = args.meta_bert_lr

        self.beta = args.beta

        self.net = Learner(args)

        self.opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.model.parameters()),
                                         lr=self.update_lr, weight_decay=0)
        self.bert_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.bert_model.parameters()),
                                              lr=self.update_bert_lr, weight_decay=0)

        self.meta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.model.parameters()),
                               lr=self.meta_lr, weight_decay=0)
        self.meta_bert_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.bert_model.parameters()),
                                    lr=self.meta_bert_lr, weight_decay=0)

    def forward(self, batch_spt, batch_qry, tables):

        losses = 0
        losses_q = 0

        loss = train_step(batch_spt, tables, self.net.model, self.net.bert_model, self.net.bert_config,
                          self.net.bert_tokenizer, self.net.ch_tokenizer,
                          self.net.max_seq_len,
                          self.net.n_layers_bert_out,
                          self.net.h_num_limit,
                          device=self.args.gpu)

        if loss is not None:
            losses += loss
            self.opt.zero_grad()
            self.bert_opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
            self.bert_opt.step()

        loss_q = train_step(batch_qry, tables, self.net.model, self.net.bert_model, self.net.bert_config,
                          self.net.bert_tokenizer, self.net.ch_tokenizer,
                          self.net.max_seq_len,
                          self.net.n_layers_bert_out,
                          self.net.h_num_limit,
                          device=self.args.gpu)
        if loss_q is not None:
            losses_q += loss_q

        meta_loss = (self.beta * losses + (1 - self.beta) * losses_q)
        self.meta_opt.zero_grad()
        self.meta_bert_opt.zero_grad()
        meta_loss.backward()
        self.meta_opt.step()
        self.meta_bert_opt.step()
        return loss