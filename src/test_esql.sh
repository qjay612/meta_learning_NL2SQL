#!/bin/bash

devices=0

python test_esql.py \
--n_op 7 \
--n_agg 6 \
--n_limit 105 \
--max_sel_num 5 \
--max_where_num 5 \
--bs 2 \
--max_seq_len 320 \
--n_layers_bert_out 2 \
--bert_path ../data/bert-base-chinese \
--ch_vocab_path ../data/esql/char_vocab.pkl \
--test_data ../data/esql/test_zs_aug.jsonl \
--test_table ../data/esql/tables.jsonl \
--cpt ./runs/esql/1607435502/checkpoints/best_snapshot_epoch_13_best_val_acc_0.813_meta_learner.pt \
--gpu $devices