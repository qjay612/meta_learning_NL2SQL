# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/3
# @Author  : Yongrui Chen
# @File    : test_esql.py
# @Software: PyCharm
"""

import os
import sys
import pickle
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append("..")
from src.utils.utils import load_bert_from_tf, load_data, save_for_evaluation
from src.utils.utils_esql import test
from src.meta.meta_esql import Meta
from src.pargs import pargs


if __name__ == "__main__":
    args = pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for testing.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for testing.\n')

    meta_learner = Meta(args)
    meta_learner.load_state_dict(torch.load(args.cpt, map_location='cpu'))

    if args.cuda:
        meta_learner.cuda()
        device = args.gpu
        print('\nShift model to GPU.')
    else:
        device = -1

    print("\nDataset: esql")
    print("Load test data from \"{}\".".format(os.path.abspath(args.test_data)))
    print("Load test tables from \"{}\".".format(os.path.abspath(args.test_table)))
    test_data, test_tables = load_data(args.test_data, args.test_table)

    if args.use_small:
        test_data = test_data[:200]

    test_loader = DataLoader(dataset=test_data, batch_size=args.bs, shuffle=False,
                              num_workers=1, collate_fn=lambda x: x)
    print("\nTest data number: {}, batch_size: {}, shuffle: {}".format(len(test_data), args.bs, False))


    print("Testing start ...")

    log_header = '\n   SCN     SC    SCA     WO' \
                 '    WCN     WC    WCA    WCO    WCV    OBO    OBC    OBL    TOTAL'
    test_log_template = ' '.join(
        '{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},'
        '{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:8.3f}'.split(','))
    print(log_header)

    with torch.no_grad():
        test_acc, results = test(test_loader, test_tables,
                                meta_learner.net.model, meta_learner.net.bert_model,
                                meta_learner.net.bert_config, meta_learner.net.bert_tokenizer,
                                meta_learner.net.ch_tokenizer,
                                max_seq_len=args.max_seq_len,
                                n_layers_bert_out=args.n_layers_bert_out, EG=args.EG,
                                h_num_limit=args.h_num_limit,
                                device=args.gpu)
    print(test_log_template.format(test_acc[0], test_acc[1],
                                  test_acc[2], test_acc[3],
                                  test_acc[4], test_acc[5],
                                  test_acc[6], test_acc[7],
                                  test_acc[8], test_acc[9],
                                  test_acc[10], test_acc[11], test_acc[12]))

    cpt_dir = '/'.join(args.cpt.split('/')[:-2])
    save_for_evaluation(cpt_dir, results, "test")

    print('\nTesting finished.')
    print("\nFinal Acc: {:.3f}\n".format(test_acc[-1]))
    print("Test results are saved to \"{}\".".format(os.path.abspath(cpt_dir)))
