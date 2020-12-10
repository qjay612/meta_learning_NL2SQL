#!/bin/bash

devices=0

python test_wikisql.py \
--bs 12 \
--n_layers_bert_out 2 \
--n_op 4 \
--n_agg 6 \
--max_where_num 4 \
--bert_path ../data/bert-base-uncased \
--ch_vocab_path ../data/wikisql/char_vocab.pkl \
--test_data ../data/wikisql/test_zs_aug.jsonl \
--test_table ../data/wikisql/test_zs.tables.jsonl \
--db_path ../data/wikisql \
--cpt ./runs/wikisql/1599269720/checkpoints/best_snapshot_epoch_3_best_val_acc_0.8408740054625341_meta_learner.pt \
--gpu $devices