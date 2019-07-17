# utils.py
# author: cyn
# email: yunachen@stu.xmu.edu.cn

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import Counter
import numpy as np

from utils.config import *



class Vocabulary():
    def __init__(self, train_file, vocab_file, vocab_addition):
        self.train_path = train_file
        self.vocab_path = vocab_file

        self.vocab_addition = vocab_addition

        self.word2id, self.id2word = self.build_vocab()

    def read_words(self, filename):
        return open(filename).read().split()

    def build_vocab(self):
        if not os.path.exists(self.vocab_path):
            with open(self.vocab_path, "w") as fvoc:
                content = self.read_words(self.train_path)
                count = Counter(content)
                sorted_count = sorted(count.items(),
                                      key=lambda x: (-x[1], x[0]))
                words, _ = list(zip(*sorted_count))
                words = (self.vocab_addition +
                         list(words))[:args.max_vocab_size]
                fvoc.write("\n".join(words))
        else:
            words = self.read_words(self.vocab_path)
        word2id = dict(zip(words, range(len(words))))
        id2word = {}
        for word in word2id:
            id2word[word2id[word]] = word

        return word2id, id2word

    def sentence2id(self, sentence):
        ids = [self.word2id.get(word, UNK) for word in sentence]
        ids.append(EOS)
        return ids

    def voc_len(self):
        return len(self.word2id)


class train_Dataset(Dataset):
    def __init__(self, src_file, tgt_file, src_voc, tgt_voc):
        with open(src_file, 'r') as f:
            src_data = f.readlines()
        with open(tgt_file, 'r') as f:
            tgt_data = f.readlines()

        self.src_data = []
        self.tgt_data = []
        for i in range(len(src_data)):
            if len(src_data[i].strip().split(' ')) <= args.max_sequen_size - 1 \
                    and len(tgt_data[i].strip().split(' ')) \
                    <= args.max_sequen_size - 1:
                self.src_data.append(src_data[i])
                self.tgt_data.append(tgt_data[i])

        self.src_voc = src_voc
        self.tgt_voc = tgt_voc

    def __getitem__(self, index):
        src_sent = self.src_voc.sentence2id(
            self.src_data[index].strip().split(' '))
        tgt_sent = self.tgt_voc.sentence2id(
            self.tgt_data[index].strip().split(' '))
        data = (src_sent, tgt_sent, len(src_sent), len(tgt_sent))
        return data

    def __len__(self):
        return len(self.src_data)


def padding(data, data_len):
    max_len = max(data_len)
    inputs = np.zeros((len(data), max_len),dtype=int)
    for i in range(len(inputs)):
        inputs[i][:len(data[i])] = data[i]
    return torch.LongTensor(inputs), torch.LongTensor(data_len)


def train_iterator(src_file, tgt_file, batch_size, sword2id, tword2id):
    dataset = train_Dataset(src_file, tgt_file, sword2id, tword2id)

    def collate_fn(batch):
        # torch.LongTensor
        src, tgt, src_len, tgt_len = zip(*batch)

        src, src_len = padding(list(src), list(src_len))
        tgt, tgt_len = padding(list(tgt), list(tgt_len))

        src_len, x_indices = torch.sort(src_len, dim=-1, descending=True)
        src = src[x_indices]
        tgt = tgt[x_indices]
        tgt_len = tgt_len[x_indices]

        return src, tgt, src_len, tgt_len

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)
    return dataloader


class eval_Dataset(Dataset):
    def __init__(self, src_file, src_voc):
        with open(src_file, 'r') as f:
            self.src_data = f.readlines()
        self.src_voc = src_voc

    def __getitem__(self, index):
        src_sent = self.src_voc.sentence2id(
            self.src_data[index].strip().split(' '))
        data = (src_sent, len(src_sent))
        return data

    def __len__(self):
        return len(self.src_data)


def eval_iterator(src_file, sword2id):
    dataset = eval_Dataset(src_file, sword2id)

    def collate_fn(batch):
        src, src_len = zip(*batch)
        return torch.LongTensor(src), torch.LongTensor(src_len)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
    return dataloader




