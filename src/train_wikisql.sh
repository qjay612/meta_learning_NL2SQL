#!/bin/bash

devices=0

python -u train_wikisql.py \
--beta 0.3 \
--seed 1 \
--bert_path ../data/bert-base-uncased \
--ch_vocab_path ../data/wikisql/char_vocab.pkl \
--train_data ../data/wikisql/train_aug.jsonl \
--train_table ../data/wikisql/train.tables.jsonl \
--dev_data ../data/wikisql/dev_aug.jsonl \
--dev_table ../data/wikisql/dev.tables.jsonl \
--db_path ../data/wikisql \
--gpu $devices \
--n_op 4 \
--n_agg 6 \
--max_where_num 4 \
--bs 8 \
--n_tasks 10000 \
--n_way 4 \
--k_shot 4 \
--update_lr 1e-3 \
--update_bert_lr 1e-5 \
--meta_lr 1e-3 \
--meta_bert_lr 1e-5 \
--n_layers_bert_out 2