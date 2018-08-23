from collections import Counter
import os


class PtbReader:

    def __init__(self, data_path):
        self.train_path = os.path.join(data_path, 'ptb.train.txt')
        self.valid_path = os.path.join(data_path, 'ptb.valid.txt')
        self.test_path = os.path.join(data_path, 'ptb.test.txt')

        self.word_to_id, self.id_to_word = self.build_vocab()

    def read_words(self, filename):
        return open(filename).read().split()

    def build_vocab(self):
        t = self.read_words(self.train_path)
    
        count = Counter(t)
        count_pairs = sorted(count.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        id_to_word = {}
        for word in word_to_id:
            id_to_word[word_to_id[word]] = word

        return word_to_id, id_to_word

    def file_to_word_ids(self, filename):
        data = self.read_words(filename)
        return [self.word_to_id[word] for word in data if word
                in self.word_to_id]

    def ptb_data(self):
        train_data = self.file_to_word_ids(self.train_path)
        valid_data = self.file_to_word_ids(self.valid_path)
        test_data = self.file_to_word_ids(self.test_path)
        vocabulary = len(self.word_to_id)
        return train_data, valid_data, test_data, vocabulary



