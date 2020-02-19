import os
import re
import numpy as np
from gensim.models import word2vec
from nltk.tokenize import TreebankWordTokenizer


def word2vector():
    # pretrained_emb_path = os.path.join(os.getcwd(),'../dataset/word_embeddings', 'glove_42B_300d_vec_plus_word2vec_100.txt')
    pretrained_emb_path = os.path.join(os.getcwd(), 'resources', 'sgns.weibo.bigram-char')
    w2v = word2vec.Word2VecKeyedVectors.load_word2vec_format(pretrained_emb_path, binary=False)
    return w2v


def stopword_remover(text):
    stop_words = [' __path__', ' __number__', ' __url__']
    pattern = '|'.join(stop_words)
    text = re.sub(pattern, '', text, count=0, flags=0)
    return text


def emb_lookup(text, w2v):
    nltk_tokenizer = TreebankWordTokenizer()
    text = nltk_tokenizer.tokenize(text)
    vector = np.zeros(w2v.vector_size)
    for word in text:
        try:
            vector += w2v[word]
        except:
            pass
    return vector


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def mrr(output, labels):
    score = output[:, 1].data
    # labels = labels.data
    # print(labels)
    corresponding = dict(zip(score, labels))
    ranking = [corresponding[k] for k in sorted(corresponding.keys(), reverse=True)]
    # print(ranking)
    # exit()
    i = 0
    for s in ranking:
        i += 1
        if s == 1:
            return i


def hit_at_n(ranking, n):
    hit = 0
    for rank in ranking:
        if rank <= n:
            hit += 1
    return hit / len(ranking)
