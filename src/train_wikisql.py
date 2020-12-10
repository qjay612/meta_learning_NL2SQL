# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : train_wikisql.py
# @Software: PyCharm
"""

import os
import sys
import time
import glob
import torch
import pickle
sys.path.append("..")
sys.stdout.flush()
import src.pargs as pargs
from torch.utils.data import  DataLoader
from src.utils.utils import load_data, load_meta_datas, save_for_evaluation, get_dataset_name
from src.meta.meta_wikisql import Meta
from src.utils.utils_wikisql import test


if __name__ == '__main__':
    args = pargs.pargs()

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('\nNote: You are using GPU for training.\n')
        torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available() and not args.cuda:
        print('\nWarning: You have Cuda but do not use it. You are using CPU for training.\n')

    meta_learner = Meta(args)

    if args.cuda:
        meta_learner.cuda()
        device = args.gpu
        print('Shift model to GPU {}.\n'.format(device))
    else:
        device = -1

    print("\nDataset: WikiSQL")
    print("Load training data from \"{}\".".format(os.path.abspath(args.train_data)))
    print("Load training tables from \"{}\".".format(os.path.abspath(args.train_table)))
    train_meta_datas, train_tables = load_meta_datas(args.train_data, args.train_table,
                                                     args.n_way, args.k_shot, args.n_tasks)

    print("Load dev data from \"{}\".".format(os.path.abspath(args.dev_data)))
    print("Load dev tables from \"{}\".".format(os.path.abspath(args.dev_table)))
    dev_datas, dev_tables = load_data(args.dev_data, args.dev_table)
    dataset_name = get_dataset_name(args.dev_data)

    if args.use_small:
        train_meta_datas = train_meta_datas[:5]
        dev_datas = dev_datas[:20]

    dev_loader = DataLoader(dataset=dev_datas, batch_size=args.bs, shuffle=False,
                            num_workers=1, collate_fn=lambda x: x)
    print("\nTraining Task Number: {}, N-way: {}, K-shot: {}".format(len(train_meta_datas),
                                                                     args.n_way, args.k_shot))
    print("Dev Data Number: {}, batch_size: {}, shuffle: False".format(len(dev_datas), args.bs))

    print("Training of {} epochs starts ...".format(args.n_epochs))

    # create runs directory.
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', "wikisql", timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('\nModel writing to \"{}\"'.format(out_dir))
    with open(os.path.join(out_dir, 'param.log'), 'w') as fin:
        fin.write(str(args))
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    iters = 0
    start = time.time()
    best_dev_lf_acc = 0

    header = '\n  Time   Epoch       Loss       SC      SCA       WN       WC       WO       WV      TOTAL'
    val_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>12.6f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))
    best_snapshot_prefix = os.path.join(checkpoint_dir, 'best_snapshot')

    print(header)

    for epoch in range(1, args.n_epochs + 1):
        cnt = 0
        avg_loss = 0
        meta_learner.train()
        for iB, (spt, qry) in enumerate(train_meta_datas):
            cnt += 1
            loss = meta_learner(spt, qry, train_tables)
            avg_loss += loss

        avg_loss /= cnt

        with torch.no_grad():
            dev_acc, results = test(dev_loader, dev_tables,
                                meta_learner.net.model, meta_learner.net.bert_model,
                                meta_learner.net.bert_config, meta_learner.net.bert_tokenizer,
                                meta_learner.net.ch_tokenizer,
                                n_layers_bert_out=args.n_layers_bert_out, EG=args.EG,
                                path_db=args.db_path,
                                beam_size=args.beam_size,
                                device=args.gpu, dset_name=dataset_name)
        print(val_log_template.format(time.time() - start, epoch, avg_loss,
                                      dev_acc[0], dev_acc[1], dev_acc[2],
                                      dev_acc[3], dev_acc[4], dev_acc[5], dev_acc[6]))

        dev_lf_acc = dev_acc[-1]
        # update checkpoint.
        if dev_lf_acc >= best_dev_lf_acc:
            best_dev_lf_acc = dev_lf_acc

            snapshot_path = best_snapshot_prefix + \
                            '_epoch_{}_best_val_acc_{}_meta_learner.pt'.format(epoch, best_dev_lf_acc)
            # save model, delete previous 'best_snapshot' files.
            torch.save(meta_learner.state_dict(), snapshot_path)

            save_for_evaluation(out_dir, results, "dev")

            for f in glob.glob(best_snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)
    print('\nTraining finished.')
    print("\nBest Acc: {:.2f}\nModel writing to \"{}\"\n".format(best_dev_lf_acc, out_dir))