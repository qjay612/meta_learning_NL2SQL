# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : pargs.py
# @Software: PyCharm
"""

import os
import torch
import random
import numpy as np
import argparse

def pargs():
    parser = argparse.ArgumentParser(description="nl2sql config")
    # Training Parameters
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument("--bs", default=16, type=int, help="Batch size")
    parser.add_argument("--ag", default=2, type=int,
                        help="accumulate_gradients, "
                             "The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--use_small', action='store_true', help='use small data', dest='use_small')
    parser.add_argument('--not_shuffle', action='store_false',
                        help='do not shuffle training data', dest='shuffle')
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--beta', default=0.5, type=float, help="beta of two losses")
    parser.add_argument('--n_way', default=4, type=int)
    parser.add_argument('--k_shot', default=4, type=int)
    parser.add_argument('--n_tasks', default=3000, type=int, help="total number of k tasks")

    # BERT Parameters
    parser.add_argument("--n_layers_bert_out", default=1, type=int,
                        help="The Number of final layers of BERT to be used in downstream task.")
    parser.add_argument('--bert_lr', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--update_bert_lr', default=1e-5, type=float, help='BERT model learning rate.')
    parser.add_argument('--meta_bert_lr', default=1e-7, type=float, help='BERT model learning rate.')
    parser.add_argument('--max_seq_len', type=int, default=222, help="start epoch index")

    # Seq-to-SQL module parameters
    parser.add_argument('--n_layers', default=2, type=int, help="The number of LSTM layers.")
    parser.add_argument('--dropout', default=0.3, type=float, help="Dropout rate.")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate.")
    parser.add_argument('--update_lr', default=1e-3, type=float, help='Learning rate.')
    parser.add_argument('--meta_lr', default=1e-5, type=float, help='Learning rate.')
    parser.add_argument("--d_h", default=100, type=int, help="The dimension of hidden vector in the seq-to-SQL module.")
    parser.add_argument("--d_in_ch", default=128, type=int, help="The dimension of input char embeddings")
    parser.add_argument("--d_f", default=32, type=int, help="The dimension of question feature embedding")
    parser.add_argument("--n_op", default=4, type=int, help="The total number of operators")
    parser.add_argument("--n_agg", default=6, type=int, help="The total number of aggregation function")
    parser.add_argument("--n_limit", default=6, type=int, help="The total number of limit in ORDER BY")
    parser.add_argument("--max_sel_num", default=6, type=int, help="The maximum of selections in one sql")
    parser.add_argument("--max_where_num", default=6, type=int, help="The maximum of conditions in one sql")
    parser.add_argument("--h_num_limit", default=22, type=int, help="The maximum of conditions in one sql")

    parser.add_argument('--EG',
                        default=False,
                        action='store_true',
                        help="If present, Execution guided decoding is used in test.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=4,
                        help="The size of beam for smart decoding")

    # data path
    parser.add_argument('--train_data', type=str, default=os.path.abspath(""))
    parser.add_argument('--train_table', type=str, default=os.path.abspath(""))
    parser.add_argument('--dev_data', type=str, default=os.path.abspath(""))
    parser.add_argument('--dev_table', type=str, default=os.path.abspath(""))
    parser.add_argument('--test_data', type=str, default=os.path.abspath(""))
    parser.add_argument('--test_table', type=str, default=os.path.abspath(""))

    parser.add_argument('--db_path', type=str, default=os.path.abspath(""))
    parser.add_argument('--bert_path', type=str, default=os.path.abspath(""))
    parser.add_argument('--ch_vocab_path', type=str, default=os.path.abspath(""))
    parser.add_argument('--cpt', type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    return args