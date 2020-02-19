import os
import re
import numpy as np
import random
import pandas as pd
import networkx
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from itertools import combinations
from summa import keywords
from .utils import word2vector, stopword_remover, emb_lookup
import dgl
import copy
from jieba import analyse
import re


class DataLoader:
    def __init__(self, args, topk=10, benchmark='douban3'):
        train_path = os.path.join(os.getcwd(), 'data', str(benchmark), 'train.csv')
        test_path = os.path.join(os.getcwd(), 'data', str(benchmark), 'test.csv')
        valid_path = os.path.join(os.getcwd(), 'data', str(benchmark), 'valid.csv')
        rows = None
        if args.debug_mode:
            rows = 100000
        self.train = self.load_data(train_path, rows).sample(frac=1)
        self.test = self.load_data(test_path, rows).sample(frac=1)
        self.valid = self.load_data(valid_path, 10).sample(frac=1)
        self.word2idx = self.build_vocab()
        self.keyword_generator(topk)

    def load_data(self, file_path, rows=None):
        data = pd.read_csv(file_path, header=None, usecols=[0, 1, 2, 3], nrows=rows,
                           names=['history', 'last_sentence', 'response', 'label'])
        return data

    def build_vocab(self):
        """
        data = self.train
        history = ''.join(list(data['history']))
        history = ''.join(re.sub('_EOS_', ' ', history).split())
        text = ''.join(list(data['last_sentence']) + list(data['response']))
        text = ''.join(text.split())
        text = text + history
        vocab = ['<UNK>'] + list(set([char for char in text]))
        word2idx = defaultdict(int)
        for i, char in enumerate(vocab):
            word2idx[char] = i
        return word2idx
        """
        data = self.train
        history = ' '.join(list(data['history']))
        history = re.sub('_EOS_', ' ', history)
        vocab = history.split()
        text = ' '.join(list(data['last_sentence'])+list(data['response']))
        vocab.extend(text.split())
        vocab = ['<UNK>']+list(set(vocab))
        word2idx = defaultdict(int)
        for i,char in enumerate(vocab):
            word2idx[char] = i
        return word2idx

    def keyword_generator(self, topk):
        """
        func: retrieve keywords from context
        """
        for data in [self.train, self.test, self.valid]:
            keyword_list = []
            for i, line in data.iterrows():
                history = ''.join(line['history'].split())
                history = ' '.join(history.split('_EOS_'))
                last_sentence = ''.join(line['last_sentence'].split())
                text = history + last_sentence
                # text = stopword_remover(text)
                # textrank_kw = keywords.keywords(text, split=True, ratio=1)    
                tfidf_kw = analyse.extract_tags(text, topK=topk)
                keyword = []
                # print(textrank_kw ,tfidf_kw)
                # for word in textrank_kw + tfidf_kw:
                for word in tfidf_kw:
                    # for sentence in history.split()+[last_sentence]:
                    #     if word != sentence:
                    keyword.extend(word.split())
                    keyword = list(set(keyword))
                keyword_list.append(keyword)

            data['keywords'] = keyword_list
    """
    def graph_builder(self, seq_len, device, data):
        # for data in [self.train, self.test, self.valid]:
        data_list = []
        for i, line in data.iterrows():
            history = line['history'].split('_EOS_')
            history = [''.join(i.split()) for i in history]
            history = [stopword_remover(i) for i in history]
            history = [i.strip() for i in history if len(i.strip()) != 0]
            history = list(set(history))
            last_sentence = ''.join(line['last_sentence'].split())
            last_sentence = stopword_remover(last_sentence)
            response = ''.join(line['response'].split())
            response = stopword_remover(response)
            keyword = line['keywords']

            # mapping node2id and labeling node_type
            node2id = {}
            idx = 0
            node2type = {}

            for _, node in enumerate(history):
                node2id[node] = idx
                node2type[node] = 'history'
                idx += 1
            node2id['ls' + last_sentence] = idx
            node2type['ls' + last_sentence] = 'last_sentence'
            idx += 1
            node2id['rp' + response] = idx
            node2type['rp' + response] = 'response'
            idx += 1
            for _, node in enumerate(keyword):
                node2id['kw' + node] = idx
                node2type['kw' + node] = 'keyword'
                idx += 1

            # connecting edges
            edges = []
            # connect keyword and sentence if keyword in sentence
            for word in keyword:
                for sentence in history:
                    if word in sentence:
                        edges.append((node2id[sentence], node2id['kw' + word]))
                if word in last_sentence:
                    edges.append((node2id['ls' + last_sentence], node2id['kw' + word]))
                if word in response:
                    edges.append((node2id['rp' + response], node2id['kw' + word]))
            # connect sentences containing same keyword
            for word in keyword:
                sentence_list = []
                # for sentence in (history+[last_sentence]+[response]):
                for sentence in history:
                    if word in sentence:
                        sentence_list.append(node2id[sentence])
                    if word in last_sentence:
                        sentence_list.append(node2id['ls' + last_sentence])
                    if word in response:
                        sentence_list.append(node2id['rp' + response])
                edges.extend([(i, j) for (i, j) in list(combinations(sentence_list, 2))])

            # mapping sentence or word to word_index
            node_feature = []
            for node in node2type:
                node = node.strip('kw').strip('ls').strip('rp')
                feature = [self.word2idx[char] for char in node]
                # do padding and cutting
                # if node2type[node] != 'keyword':
                feature = feature[:seq_len]
                feature.extend((seq_len - len(feature)) * [0])
                node_feature.append(feature)
            node_feature = torch.LongTensor(node_feature).to(device)

            # building graph
            g = dgl.DGLGraph()
            g.add_nodes(idx)
            # print(node2id)
            if edges:
                src, dst = tuple(zip(*edges))
                g.add_edges(src, dst)
                g.add_edges(dst, src)

            g.ndata['node_emb'] = node_feature
            label = line['label']
            data_each_graph = (g, list(node2type.values()), label)
            data_list.append(data_each_graph)
        return data_list
    """
    def graph_builder(self, seq_len, device, data):                
        # for data in [self.train, self.test, self.valid]:
        data_list = []
        for i, line in data.iterrows():
            history = line['history'].split('_EOS_')
            # history = [stopword_remover(i) for i in history]
            # history = [''.join(i.split()) for i in history]
            history = [i.strip() for i in history if len(i.strip()) != 0]
            history = list(set(history))
            # last_sentence = ''.join(line['last_sentence'].split())
            last_sentence = line['last_sentence']
            # last_sentence = stopword_remover(last_sentence)
            # response = ''.join(line['response'].split())
            response = line['response']
            # response = stopword_remover(response)
            keyword = line['keywords']

            # mapping node2id and labeling node_type
            node2id = {}
            idx = 0
            node2type = {}

            for _, node in enumerate(history):
                node2id[node] = idx
                node2type[node]='history'
                idx += 1
            node2id['ls'+last_sentence] = idx
            node2type['ls'+last_sentence]='last_sentence'
            idx += 1
            node2id['rp'+response] = idx
            node2type['rp'+response]='response'
            idx += 1
            for _, node in enumerate(keyword):
                node2id['kw'+node] = idx
                node2type['kw'+node]='keyword'
                idx += 1


            # connecting edges
            edges = []
            # connect keyword and sentence if keyword in sentence
            for word in keyword:
                for sentence in history:
                    if word in ''.join(sentence.split()):
                        edges.append((node2id[sentence], node2id['kw'+word]))
                if word in ''.join(last_sentence.split()):
                    edges.append((node2id['ls'+last_sentence], node2id['kw'+word]))
                if word in ''.join(response.split()):
                    edges.append((node2id['rp'+response], node2id['kw'+word]))
            # connect sentences containing same keyword
            for word in keyword:
                sentence_list = []
                # for sentence in (history+[last_sentence]+[response]):
                for sentence in history:
                    if word in ''.join(sentence.split()):
                        sentence_list.append(node2id[sentence])
                if word in ''.join(last_sentence.split()):
                    sentence_list.append(node2id['ls'+last_sentence])
                if word in ''.join(response.split()):
                    sentence_list.append(node2id['rp'+response])
                edges.extend([(i,j) for (i,j) in list(combinations(sentence_list, 2))])
            # adding self circle
            for i in range(idx):
                edges.append((i,i))

            # mapping sentence or word to word_index
            node_feature = []
            for node in node2type:
                node = node.strip('kw').strip('ls').strip('rp')
                node = node.split()
                feature = [self.word2idx[char] for char in node]
                # do padding and cutting
                # if node2type[node] != 'keyword':
                feature = feature[:seq_len]
                feature.extend((seq_len-len(feature))*[0])
                node_feature.append(feature)
            node_feature = torch.LongTensor(node_feature).to(device)

            # building graph
            g = dgl.DGLGraph()
            g.add_nodes(idx)
            # print(node2id)
            # print(edges)
            if edges:
                src, dst = tuple(zip(*edges))
                g.add_edges(src, dst)
                g.add_edges(dst, src)
            g.ndata['node_emb'] = node_feature
            label = line['label']
            data_each_graph = (g,list(node2type.values()),label)
            data_list.append(data_each_graph)
        return data_list


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader()
    # print(data_loader.train)
    data_loader.graph_builder(10, device, data_loader.train)
