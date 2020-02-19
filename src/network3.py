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


import torch
from .modules import Module, ModuleList, ModuleDict
from .modules.embedding import Embedding, BertEmbedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import Pooling
from .modules.prediction import registry as prediction
from .modules.graph import RnnGcn


class Network(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        if args.pretrained_mode == "bert":
            self.embedding = BertEmbedding(args)
        else:
            self.embedding = Embedding(args)
        # self.gcn = RnnGcn(64, 128)
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, 400 if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, 400 + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, 400 + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, inputs):
        """
        :param inputs: shape of a, b, mask_a, mask_b (batch size * text length)
        :return: output: (batch size * num of labels)
        """
        c = inputs['context']
        q = inputs['query']
        r = inputs['response']
        mask_c = inputs['context mask']
        mask_q = inputs['query mask']
        mask_r = inputs['response mask']
        graph = inputs["graph"]
        # graph_output = self.gcn(graph)
        # graph_output = torch.unsqueeze(graph_output, 1)
        # batch size * text length * embedding hidden size
        c = self.embedding(c)
        q = self.embedding(q)
        r = self.embedding(r)
        # c = torch.cat([c, graph_output.repeat(1, list(c.size())[1], 1)], 2)
        # q = torch.cat([q, graph_output.repeat(1, list(q.size())[1], 1)], 2)
        # r = torch.cat([r, graph_output.repeat(1, list(r.size())[1], 1)], 2)
        res_c, res_q, res_r = c, q, r

        for i, block in enumerate(self.blocks):
            if i > 0:
                c = self.connection(c, res_c, i)
                q = self.connection(q, res_q, i)
                r = self.connection(r, res_r, i)
                res_c, res_q, res_r = c, q, r
            # batch size * text length * encoder size
            c_enc = block['encoder'](c, mask_c)
            q_enc = block['encoder'](q, mask_q)
            r_enc = block['encoder'](r, mask_r)
            # batch size * text length * (embedding size/connection hidden size + encoder hidden s ize)
            c = torch.cat([c, c_enc], dim=-1)
            q = torch.cat([q, q_enc], dim=-1)
            r = torch.cat([r, r_enc], dim=-1)
            align_qc, align_cq = block['alignment'](q, c, mask_q, mask_c)
            align_rc, align_cr = block['alignment'](r, c, mask_r, mask_c)
            align_q, align_r = block['alignment'](align_qc, align_rc, mask_q, mask_r)
            c = block['fusion'](c, align_cq)
            q = block['fusion'](q, align_q)
            r = block['fusion'](r, align_r)
        # before pooling shape of a, b (batch size * text length * hidden size)
        q = self.pooling(q, mask_q)
        r = self.pooling(r, mask_r)
        return self.prediction(q, r)
