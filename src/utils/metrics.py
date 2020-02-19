# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess

import operator
import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict
from functools import partial
from .registry import register

registry = {}
register = partial(register, registry=registry)


@register('acc')
def acc(outputs):
    target = outputs['target']
    pred = outputs['pred']
    score = metrics.accuracy_score(target, pred).item()
    return {
        'acc': score,
    }


@register('f1')
def f1(outputs):
    target = outputs['target']
    pred = outputs['pred']
    score = metrics.f1_score(target, pred).item()
    return {
        'f1': score,
    }


@register('auc')
def auc(outputs):
    target = outputs['target']
    prob = np.array(outputs['prob'])
    return {
        'auc': metrics.roc_auc_score(target, prob[:, 1]).item(),
    }


# @register('mrr')
# def mrr(outputs):
#     probs = np.array(outputs["prob"])
#     tagets = outputs["target"]
#     q_ids = outputs["q_ids"]
#     a_ids = outputs["a_ids"]
#     result = pd.DataFrame({'probs': probs[:, 1], 'targets': tagets, 'q_ids': q_ids, 'a_ids': a_ids})
#     result["rank"] = result["probs"].groupby(result["q_ids"]).rank(ascending=False)
#     result["rec_rank"] = result["rank"].rdiv(1)
#     mrr = result[result["targets"] == 1]["rec_rank"].sum() / (result[result["targets"] == 1].shape[0])
#     mrr = float(mrr)
#     return {'mrr': mrr}


def is_valid_query(each_answer):
    # 计算指标的时候对答案标签的合法性进行判断避免除0
    num_pos = 0
    num_neg = 0
    for label, score in each_answer:
        if label > 0:
            num_pos += 1
        else:
            num_neg += 1
    if num_pos > 0 and num_neg > 0:
        return True
    else:
        return False


@register('map')
@register('mrr')
def compute_douban(outputs):
    ID = outputs["q_ids"]
    scores = np.array(outputs["prob"])
    labels = outputs["target"]
    assert len(ID) == scores.shape[0] == len(labels)
    MRR, num_query = 0, 0
    results = defaultdict(list)
    predict = pd.DataFrame({'scores': scores[:, 1], 'labels': labels, 'ID': ID})
    for index, row in predict.iterrows():
        results[row[2]].append((row[1], row[0]))

    for key, value in results.items():
        # if not is_valid_query(value):
        #     continue
        num_query += 1
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        for index_, final_result in enumerate(sorted_result):
            label, scores = final_result
            if label > 0:
                MRR += 1.0 / (index_ + 1)
                break

    predict['rank'] = predict['scores'].groupby(predict['ID']).rank(ascending=False)
    predict['rec_rank'] = predict['rank'].rdiv(1)
    mrr = predict[predict['labels'] == 1]['rec_rank'].sum() / (predict[predict['labels'] == 1].shape[0])

    MAP = 0
    for key, value in results.items():
        if not is_valid_query(value): continue
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        num_relevant_resp = 0
        AVP = 0  # 每个文档的平均准确率
        for index_, final_result in enumerate(sorted_result):
            each_label, each_score = final_result
            if each_label > 0:
                num_relevant_resp += 1
                precision = num_relevant_resp / (index_ + 1)
                AVP += precision
        AVP = AVP / num_relevant_resp
        MAP += AVP

    Precision_1 = 0
    for key, value in results.items():
        if not is_valid_query(value):
            continue
        sorted_result = sorted(value, key=operator.itemgetter(1), reverse=True)
        # 预测的label取最后概率向量里面最大的那一个作为预测结果
        label, score = sorted_result[0]
        if label > 0:
            Precision_1 += 1

    return {
        'map': MAP / num_query,
        'mrr': MRR / num_query,
        'p@1': Precision_1 / num_query
    }

# @register('map')
# @register('mrr')
# def ranking(outputs):
#     args = outputs['args']
#     prediction = [o[1] for o in outputs['prob']]
#     ref_file = os.path.join(args.data_dir, '{}.ref'.format(args.eval_file))
#     rank_file = os.path.join(args.data_dir, '{}.rank'.format(args.eval_file))
#     tmp_file = os.path.join(args.summary_dir, 'tmp-pred.txt')
#     with open(rank_file) as f:
#         prefix = []
#         for line in f:
#             prefix.append(line.strip().split())
#         assert len(prefix) == len(prediction), \
#             'prefix {}, while prediction {}'.format(len(prefix), len(prediction))
#     with open(tmp_file, 'w') as f:
#         for prefix, pred in zip(prefix, prediction):
#             prefix[-2] = str(pred)
#             f.write(' '.join(prefix) + '\n')
#     sp = subprocess.Popen('./resources/trec_eval {} {} | egrep "map|recip_rank"'.format(ref_file, tmp_file),
#                           shell=True,
#                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = sp.communicate()
#     stdout, stderr = stdout.decode(), stderr.decode()
#     os.remove(tmp_file)
#     map_, mrr = [float(s[-6:]) for s in stdout.strip().split('\n')]
#     return {
#         'map': map_,
#         'mrr': mrr,
#     }
