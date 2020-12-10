#!/bin/bash

devices=0

python -u train_esql.py \
--beta 0.5 \
--seed 1 \
--bert_path ../data/bert-base-chinese \
--ch_vocab_path ../data/esql/char_vocab.pkl \
--train_data ../data/esql/train_aug.jsonl \
--train_table ../data/esql/tables.jsonl \
--dev_data ../data/esql/dev_aug.jsonl \
--dev_table ../data/esql/tables.jsonl \
--gpu $devices \
--n_op 7 \
--n_agg 6 \
--n_limit 105 \
--max_sel_num 5 \
--max_where_num 5 \
--bs 8 \
--n_tasks 1000 \
--n_way 1 \
--k_shot 4 \
--max_seq_len 320 \
--update_lr 1e-3 \
--update_bert_lr 1e-5 \
--meta_lr 1e-3 \
--meta_bert_lr 1e-5 \
--n_layers_bert_out 2