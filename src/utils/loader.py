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
import numpy as np


def load_data(data_dir, split=None, debug=False, mode="none"):
    data = []
    if split is None:
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
    else:
        if not split.endswith('.txt'):
            split += '.txt'
        files = [os.path.join(data_dir, f'{split}')]
    for file in files:
        with open(file) as f:
            for id, line in enumerate(f):
                if id == 10000 and debug:
                    break
                if mode == "ubuntu" or mode == "douban":
                    # 数据格式为：text1_id \t text1 \t text2_id \t text2 \t target
                    sample = line.rstrip().split('\t')
                    assert len(sample) == 5, f"length of sample: {len(sample)}"
                    text1_id, text1, text2_id, text2, target = sample
                    data.append({
                        'text1_id': text1_id,
                        'text1': text1,
                        'text2_id': text2_id,
                        'text2': text2,
                        'target': target,
                    })

                elif mode == "douban3":
                    query_id, context, query, response_id, response, label = line.rstrip().split('\t')
                    # stopwords = [line.strip() for line in open(os.path.join(data_dir, 'stopwords.txt'), encoding='UTF-8').readlines()]
                    # context = " ".join([word for word in context.split() if word not in stopwords])
                    # query = " ".join([word for word in query.split() if word not in stopwords])
                    # response = " ".join([word for word in response.split() if word not in stopwords])
                    data.append({
                        'query id': query_id,
                        'context': context,
                        'query': query,
                        'response id': response_id,
                        'response': response,
                        'target': label,
                    })
                else:
                    query, response, label = line.rstrip().split('\t')
                    data.append({
                        'query': query,
                        'response': response,
                        'target': label,
                    })
    return data


def load_embeddings(file, vocab, dim, lower, mode='freq'):
    embedding = np.zeros((len(vocab), dim))
    count = np.zeros((len(vocab), 1))
    with open(file) as f:
        for line in f:
            elems = line.rstrip().split()
            if len(elems) != dim + 1:
                continue
            token = elems[0]
            if lower and mode != 'strict':
                token = token.lower()
            if token in vocab:
                index = vocab.index(token)
                vector = [float(x) for x in elems[1:]]
                if mode == 'freq' or mode == 'strict':
                    if not count[index]:
                        embedding[index] = vector
                        count[index] = 1.
                elif mode == 'last':
                    embedding[index] = vector
                    count[index] = 1.
                elif mode == 'avg':
                    embedding[index] += vector
                    count[index] += 1.
                else:
                    raise NotImplementedError('Unknown embedding loading mode: ' + mode)
    if mode == 'avg':
        inverse_mask = np.where(count == 0, 1., 0.)
        embedding /= count + inverse_mask
    return embedding.tolist()
