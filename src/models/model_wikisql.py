# -*- coding: utf-8 -*-
# !/usr/bin/python

"""
# @Time    : 2020/8/1
# @Author  : Yongrui Chen
# @File    : model_wikisql.py
# @Software: PyCharm
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import argmax, array, zeros

sys.path.append("..")
from src.models.modules import SelectColumn, SelectAggregation, WhereNumber, WhereColumn, WhereOperator, WhereValue
from src.utils.utils import pred_where_num, pred_where_op, pred_where_col, pred_where_val_beam
from src.utils.utils import pred_sel_col_for_beam, topk_multi_dim, remap_sc_idx, check_sc_sa_pairs
from src.utils.utils import convert_val_index_wp_to_string
from src.utils.utils_wikisql import merge_val_to_english


class Seq2SQL(nn.Module):
    def __init__(self, d_in, d_in_ch, d_h, d_f, n_layers, dropout_prob, ch_vocab_size,
                 n_op, n_agg, max_where_num=4, pooling_type="last"):
        super(Seq2SQL, self).__init__()
        self.d_in = d_in
        self.d_in_ch = d_in_ch
        self.d_h = d_h
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob

        self.max_where_num = max_where_num
        self.n_op = n_op
        self.n_agg = n_agg

        self.embed_ch = nn.Embedding(ch_vocab_size, d_in_ch)

        self.m_sel_col = SelectColumn(d_in, d_h, n_layers, dropout_prob, pooling_type)
        self.m_sel_agg = SelectAggregation(d_in, d_h, n_layers, dropout_prob, n_agg, pooling_type, max_sel_num=1)
        self.m_where_num = WhereNumber(d_in, d_in_ch, d_h, n_layers, dropout_prob, pooling_type, max_where_num)
        self.m_where_col = WhereColumn(d_in, d_in_ch, d_h, n_layers, dropout_prob, pooling_type)
        self.m_where_op = WhereOperator(d_in, d_h, n_layers, dropout_prob, n_op, pooling_type, max_where_num)
        self.m_where_val = WhereValue(d_in, d_in_ch, d_h, d_f, n_layers, dropout_prob, n_op, pooling_type, max_where_num)

    def forward(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch, q_feature,
                gold_sel_col=None, gold_where_num=None, gold_where_col=None, gold_where_op=None):

        # select columns
        score_sel_col = self.m_sel_col(q_emb, q_lens, h_emb, h_lens, h_nums)

        if gold_sel_col:
            now_sel_col = gold_sel_col
        else:
            now_sel_col = pred_where_num(score_sel_col)

        # select aggregation
        score_sel_agg = self.m_sel_agg(q_emb, q_lens, h_emb, h_lens, h_nums, now_sel_col)

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

        return score_sel_col, score_sel_agg, score_where_num, \
               score_where_col, score_where_op, score_where_val

    def beam_forward(self, q_emb, q_lens, h_emb, h_lens, h_nums,
                     q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch, q_feature,
                     engine, tb, q_tok_cn, q_tok_wp, wp_to_cn_index, q,
                     beam_size=4, device=-1):
        """
        Execution-guided beam decoding.
        """
        # sc
        s_sc = self.m_sel_col(q_emb, q_lens, h_emb, h_lens, h_nums)
        prob_sc = F.softmax(s_sc, dim=-1)
        bs, mcL = s_sc.shape

        # minimum_hs_length = min(h_nums)
        # beam_size = minimum_hs_length if beam_size > minimum_hs_length else beam_size

        # sa
        # Construct all possible sc_sa_score
        prob_sc_sa = torch.zeros([bs, beam_size, self.n_agg])
        prob_sca = torch.zeros_like(prob_sc_sa)
        if device != -1:
            prob_sc_sa = prob_sc_sa.to(device)
            prob_sca = prob_sca.to(device)

        # get the top-k indices.  pr_sc_beam = [B, beam_size]
        pr_sc_beam = pred_sel_col_for_beam(s_sc, beam_size)

        # calculate and predict s_sa.
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.m_sel_agg(q_emb, q_lens, h_emb, h_lens, h_nums, pr_sc)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa

            prob_sc_selected = prob_sc[range(bs), pr_sc]  # [B]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
            # [mcL, B] * [B] -> [mcL, B] (element-wise multiplication)
            # [mcL, B] -> [B, mcL]

        # Calculate the dimension of tensor
        # tot_dim = len(prob_sca.shape)

        # First flatten to 1-d
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        # Now as sc_idx is already sorted, re-map them properly.

        idxs = remap_sc_idx(idxs, pr_sc_beam)  # [sc_beam_idx, sa_idx] -> [sc_idx, sa_idx]
        idxs_arr = array(idxs)
        # [B, beam_size, remainig dim]
        # idxs[b][0] gives first probable [sc_idx, sa_idx] pairs.
        # idxs[b][1] gives of second.

        # Calculate prob_sca, a joint probability
        beam_idx_sca = [0] * bs
        beam_meet_the_final = [False] * bs
        while True:
            pr_sc = idxs_arr[range(bs), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bs), beam_idx_sca, 1]

            # map index properly
            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)

            if sum(check) == bs:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:  # wrong pair
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True

            if sum(beam_meet_the_final) == bs:
                break

        # Now pr_sc, pr_sa are properly predicted.
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)

        # Now, Where-clause beam search.
        s_wn = self.m_where_num(q_emb, q_lens, h_emb, h_lens, h_nums,
                                q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch)
        prob_wn = F.softmax(s_wn, dim=-1).detach().to('cpu').numpy()

        # Found "executable" most likely 4(=max_num_of_conditions) where-clauses.
        # wc
        s_wc = self.m_where_col(q_emb, q_lens, h_emb, h_lens, h_nums,
                                q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch)
        prob_wc = F.sigmoid(s_wc).detach().to('cpu').numpy()
        # pr_wc_sorted_by_prob = pred_wc_sorted_by_prob(s_wc)

        # get max_wn # of most probable columns & their prob.
        pr_wn_max = [self.max_where_num] * bs
        pr_wc_max = pred_where_col(pr_wn_max,
                                   s_wc)  # if some column do not have executable where-claouse, omit that column
        prob_wc_max = zeros([bs, self.max_where_num])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]

        # get most probable max_wn where-clouses
        # wo
        s_wo_max = self.m_where_op(q_emb, q_lens, h_emb, h_lens, h_nums, pr_wc_max)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().to('cpu').numpy()
        # [B, max_wn, n_cond_op]

        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op in range(self.n_op - 1):
            pr_wo_temp = [[i_op] * self.max_where_num] * bs
            # wv
            s_wv = self.m_where_val(q_emb, q_lens, h_emb, h_lens, h_nums,
                                    q_emb_ch, q_lens_ch, h_emb_ch, h_lens_ch,
                                    pr_wc_max, pr_wo_temp,
                                    q_feature)
            prob_wv = F.softmax(s_wv, dim=-2).detach().to('cpu').numpy()

            # prob_wv
            pr_wvi_beam, prob_wvi_beam = pred_where_val_beam(self.max_where_num, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
            # pr_wvi_beam = [B, max_wn, k_logit**2 [st, ed] paris]

            # pred_wv_beam

        # Calculate joint probability of where-clause
        # prob_w = [batch, wc, wo, wv] = [B, max_wn, n_cond_op, n_pairs]
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bs, self.max_where_num, self.n_op - 1, n_wv_beam_pairs])
        for b in range(bs):
            for i_wn in range(self.max_where_num):
                for i_op in range(self.n_op - 1):  # do not use final one
                    for i_wv_beam in range(n_wv_beam_pairs):
                        # i_wc = pr_wc_max[b][i_wn] # already done
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]

                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv

        # Perform execution guided decoding
        conds_max = []
        prob_conds_max = []
        # while len(conds_max) < self.max_where_num:
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        # idxs = [B, i_wc_beam, i_op, i_wv_pairs]

        # Construct conds1
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]

                # get wv_str
                temp_pr_wv_str, _ = convert_val_index_wp_to_string([wp_to_cn_index[b]], [[wvi]], [q_tok_cn[b]],
                                                                   [q_tok_wp[b]])
                merged_wv11 = merge_val_to_english(temp_pr_wv_str[0][0], q[b])
                conds11 = [i_wc, i_op, merged_wv11]

                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]

                # test execution
                # print(q[b])
                # print(tb[b]['id'], tb[b]['types'], pr_sc[b], pr_sa[b], [conds11])
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    # pr_ans is not empty!
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)

            # May need to do more exhuastive search?
            # i.e. up to.. getting all executable cases.

        # Calculate total probability to decide the number of where-clauses
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []

        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])  # wn=0 case.
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)

            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        # s_wv = [B, max_wn, max_nlu_tokens, 2]
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i