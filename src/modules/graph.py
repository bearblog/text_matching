import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
from .layers import GCN, NodeApplyModule, Rnn

class RnnGcn(nn.Module):
    def __init__(self, rnn_hidden_dim, gnn_hidden_dim):
        super(RnnGcn, self).__init__()

        self.embedding = None
        self.rnn = nn.GRU(
            input_size=300,
            hidden_size=rnn_hidden_dim,
            batch_first=True,
            bidirectional=True)
        self.gcn_layers = nn.ModuleList([
            GCN(rnn_hidden_dim * 2, gnn_hidden_dim, F.relu),
            GCN(gnn_hidden_dim, gnn_hidden_dim, F.relu),
            GCN(gnn_hidden_dim, gnn_hidden_dim, F.relu),
            # GCN(gnn_hidden_dim, gnn_hidden_dim, F.relu),
            # GCN(gnn_hidden_dim, gnn_hidden_dim, F.relu),
            # GCN(gnn_hidden_dim, gnn_hidden_dim, F.relu),
            GCN(gnn_hidden_dim, gnn_hidden_dim, F.relu)])
        # self.kw_alignment = nn.Linear(embedding_matrix.size()[1], rnn_hidden_dim*2)
        self.linear = nn.Linear(gnn_hidden_dim, 2)

    def set_embedding(self, embedding_matrix):
        self.embedding = nn.Embedding(embedding_matrix.size()[0], embedding_matrix.size()[1], _weight=embedding_matrix)

    def forward(self, batched_graph, batched_type_list):
        batched_graph = dgl.batch(batched_graph)
        batched_graph.ndata['node_emb'] = self.embedding(batched_graph.ndata['node_emb'])
        batched_graph.ndata['node_emb'], _ = self.rnn(batched_graph.ndata['node_emb'])
        batched_graph.ndata['node_emb'] = batched_graph.ndata['node_emb'][:, -1, :]
        for conv in self.gcn_layers:
            batched_graph.ndata['node_emb'] = conv(batched_graph, batched_graph.ndata['node_emb'])
        batched_graph = dgl.unbatch(batched_graph)
        # graph_repr = dgl.mean_nodes(batched_graph, 'node_emb')
        # return graph_repr
        batched_hist = []
        batched_last = []
        batched_resp = []
        for idx in range(len(batched_graph)):
            graph = batched_graph[idx]
            node_emb = graph.ndata['node_emb']
            type_list = batched_type_list[idx]
            hist_emb = []
            # last_sent_emb = []
            # response_emb = []
            for node_idx in range(len(type_list)):
                if type_list[node_idx] == 'history':
                    hist_emb.append(node_emb[node_idx])
                elif type_list[node_idx] == 'last_sentence':
                    last_sent_emb = node_emb[node_idx]
                elif type_list[node_idx] == 'response':
                    response_emb = node_emb[node_idx]
            hist_emb = torch.stack(hist_emb, 0)
            hist_emb = torch.mean(hist_emb,dim=0,keepdim=False)
            # last_sent_emb = torch.stack(last_sent_emb, 0)
            # response_emb = torch.stack(response_emb, 0)
            # print(hist_emb.size(),last_sent_emb.size(), response_emb.size())
            # exit()
            batched_hist.append(hist_emb)
            batched_last.append(last_sent_emb)
            batched_resp.append(response_emb)
        batched_hist = torch.stack(batched_hist, 0)
        batched_last = torch.stack(batched_last, 0)
        batched_resp = torch.stack(batched_resp, 0)
        # print(batched_hist.size(),batched_last.size(), batched_resp.size())
        # exit()            
        return batched_hist, batched_last, batched_resp
