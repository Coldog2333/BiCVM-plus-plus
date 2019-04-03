import os
import random
import datetime
import gensim
import psutil
import torch
import torch.utils.data
"""
2019.04.01 11:48 简化代码.
"""
embedding_dim = 300

def load_sentences(corpus):
    sentences = []
    with open(corpus, 'r', encoding='utf8') as f:
        raw = f.read().splitlines()
    for r in raw:
        sentences.append(r.split(' '))
    return sentences


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, corpus0, corpus1, word2vec0='../word2vec/en/en.bin', word2vec1='../word2vec/de/de.bin', num_of_noise=3, OOV_strategy='random'):
        self.num_of_noise = num_of_noise            # 对比句子数
        self.OOV_strategy = OOV_strategy

        # list of sentence in corpus, [0] for en-corpus, [1] for de-corpus
        self.sentences = [load_sentences(corpus0), load_sentences(corpus1)]

        # word2vec model
        self.word2vec = [gensim.models.word2vec.Word2Vec.load(word2vec0), gensim.models.word2vec.Word2Vec.load(word2vec1)]

        # dictionary of oov words
        self.oov_embedding_dict = [dict(), dict()]              # out-of-vocabulary词的词向量


    def word2vec_embedding(self, word, lang, OOV_strategy):     # lang = 0/1
        try:
            return torch.FloatTensor(self.word2vec[lang].wv.__getitem__(word)).view(1, embedding_dim), True
        except:
            if OOV_strategy == 'random':
                if word not in self.oov_embedding_dict[lang].keys():
                    self.oov_embedding_dict[lang][word] = torch.rand((1, embedding_dim))  # record the random embedding of oov word
                return self.oov_embedding_dict[lang][word], True
            elif OOV_strategy == 'cast':
                return None, False
            elif OOV_strategy == 'zero':
                return torch.zeros((1, embedding_dim)), True

    def sentence2matrix(self, sentence, lang, OOV_strategy):
        matrix = torch.zeros(1, embedding_dim)
        for i in range(len(sentence)):
            embedding, flag = self.word2vec_embedding(sentence[i], lang=lang, OOV_strategy=OOV_strategy)
            if flag == True:
                matrix = torch.cat((matrix, embedding), dim=0)
            else:
                continue
        return matrix[1:, :]

    def __getitem__(self, index):
        input0 = self.sentence2matrix(self.sentences[0][index], lang=0, OOV_strategy=self.OOV_strategy)
        input1 = self.sentence2matrix(self.sentences[1][index], lang=1, OOV_strategy=self.OOV_strategy)

        disturb_input = []
        for i in range(self.num_of_noise):
            r = random.randint(0, self.__len__() - 1)
            disturb_input.append(self.sentence2matrix(self.sentences[1][r], lang=1, OOV_strategy=self.OOV_strategy))

        return input0, input1, disturb_input

    def __len__(self):
        return len(self.sentences[0])


class CorpusLoader(torch.utils.data.Dataset):
    def __init__(self, corpus0, corpus1, word2vec0='../word2vec/en/en.bin', word2vec1='../word2vec/de/de.bin', num_of_noise=3, OOV_strategy='random'):
        self.num_of_noise = num_of_noise
        self.OOV_strategy = OOV_strategy

        self.sentences = [load_sentences(corpus0), load_sentences(corpus1)]

        self.word2vec = [gensim.models.word2vec.Word2Vec.load(word2vec0), gensim.models.word2vec.Word2Vec.load(word2vec1)]

        self.oov_embedding_dict = [dict(), dict()]              # out-of-vocabulary词的词向量

        print('Initializing sentence matrices...')
        self.matrix0 = self.initial_matrix(lang=0, OOV_strategy=self.OOV_strategy)
        self.matrix1 = self.initial_matrix(lang=1, OOV_strategy=self.OOV_strategy)
        print('Matrices are ready. Sample size = %d' % len(self.sentences[0]))

    def word2vec_embedding(self, word, lang, OOV_strategy):     # lang = 0/1
        try:
            return torch.FloatTensor(self.word2vec[lang].wv.__getitem__(word)).view(1, embedding_dim), True
        except:
            if OOV_strategy == 'random':
                if word not in self.oov_embedding_dict[lang].keys():
                    self.oov_embedding_dict[lang][word] = torch.rand((1, embedding_dim))
                return self.oov_embedding_dict[lang][word], True
            elif OOV_strategy == 'cast':
                return None, False
            elif OOV_strategy == 'zero':
                return torch.zeros((1, embedding_dim)), True

    def sentence2matrix(self, sentence, lang, OOV_strategy):
        matrix = torch.zeros(1, embedding_dim)
        for i in range(len(sentence)):
            embedding, flag = self.word2vec_embedding(sentence[i], lang=lang, OOV_strategy=OOV_strategy)
            if flag == True:
                matrix = torch.cat((matrix, embedding), dim=0)
            else:
                continue
        return matrix[1:, :]

    def initial_matrix(self, lang, OOV_strategy):
        matrix = []
        for index in range(self.__len__()):
            matrix.append(self.sentence2matrix(self.sentences[lang][index], lang=lang, OOV_strategy=OOV_strategy))

            if index / 1000 == index // 1000:
                print('Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                      (index / self.__len__() * 100, psutil.virtual_memory().percent, psutil.cpu_percent(1)))
        return matrix

    def __getitem__(self, index):
        input0 = self.matrix0[index]
        input1 = self.matrix1[index]

        disturb_input = []
        for i in range(self.num_of_noise):
            r = random.randint(0, self.__len__() - 1)
            disturb_input.append(self.matrix1[r])

        return input0, input1, disturb_input

    def __len__(self):
        return len(self.sentences[0])


class MemoryFriendlyLoader4SA(torch.utils.data.Dataset):
    def __init__(self, SAdir='../data/aclImdb_v1/aclImdb/train/', word2vec='../word2vec/en/en.bin', cut=200, OOV_strategy='random'):
        self.OOV_strategy = OOV_strategy
        self.sentences = []
        # pos
        for p in os.listdir(os.path.join(SAdir, 'pos')):
            with open(os.path.join(SAdir, 'pos', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:cut])

        self.pos_no = len(self.sentences)       # 小于self.pos_no的都是pos
        # neg
        for p in os.listdir(os.path.join(SAdir, 'neg')):
            with open(os.path.join(SAdir, 'neg', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:cut])

        self.word2vec = gensim.models.word2vec.Word2Vec.load(word2vec)
        self.oov_embedding_dict = dict()   # out-of-vocabulary词的词向量


    def word2vec_embedding(self, word, OOV_strategy):
        try:
            return torch.FloatTensor(self.word2vec.wv.__getitem__(word)).view(1, embedding_dim), True
        except:
            if OOV_strategy == 'random':
                if word not in self.oov_embedding_dict.keys():
                    self.oov_embedding_dict[word] = torch.rand((1, embedding_dim))
                return self.oov_embedding_dict[word], True
            elif OOV_strategy == 'cast':
                return None, False
            elif OOV_strategy == 'zero':
                return torch.zeros((1, embedding_dim)), True

    def sentence2matrix(self, sentence, OOV_strategy):
        matrix = torch.zeros(1, embedding_dim)
        for i in range(len(sentence)):
            embedding, flag = self.word2vec_embedding(sentence[i], OOV_strategy=OOV_strategy)
            if flag == True:
                matrix = torch.cat((matrix, embedding), dim=0)
            else:
                continue
        return matrix[1:, :]


    def __getitem__(self, index):
        input = self.sentence2matrix(self.sentences[index], OOV_strategy=self.OOV_strategy)

        if index < self.pos_no:
            ground_truth = torch.FloatTensor([1, 0])
        else:
            ground_truth = torch.FloatTensor([0, 1])

        return input, ground_truth

    def __len__(self):
        return len(self.sentences)


class CorpusLoader4SA(torch.utils.data.Dataset):
    def __init__(self, SAdir='../data/aclImdb_v1/aclImdb/train/', word2vec='../word2vec/en/en.bin', cut=200, OOV_strategy='random'):
        self.OOV_strategy = OOV_strategy
        self.sentences = []
        # pos
        for p in os.listdir(os.path.join(SAdir, 'pos')):
            with open(os.path.join(SAdir, 'pos', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:cut])

        self.pos_no = len(self.sentences)       # 小于self.pos_no的都是pos
        # neg
        for p in os.listdir(os.path.join(SAdir, 'neg')):
            with open(os.path.join(SAdir, 'neg', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:cut])

        self.word2vec = gensim.models.word2vec.Word2Vec.load(word2vec)
        self.oov_embedding_dict = dict()   # out-of-vocabulary词的词向量

        print('Initializing sentence matrices...')
        self.matrix = self.initial_matrix()
        print('Matrices are ready. Sample size = %d' % len(self.sentences))

    def word2vec_embedding(self, word, OOV_strategy):
        try:
            return torch.FloatTensor(self.word2vec.wv.__getitem__(word)).view(1, embedding_dim), True
        except:
            if OOV_strategy == 'random':
                if word not in self.oov_embedding_dict.keys():
                    self.oov_embedding_dict[word] = torch.rand((1, embedding_dim))
                return self.oov_embedding_dict[word], True
            elif OOV_strategy == 'cast':
                return None, False
            elif OOV_strategy == 'zero':
                return torch.zeros((1, embedding_dim)), True

    def sentence2matrix(self, sentence, OOV_strategy):
        matrix = torch.zeros(1, embedding_dim)
        for i in range(len(sentence)):
            embedding, flag = self.word2vec_embedding(sentence[i], OOV_strategy=OOV_strategy)
            if flag == True:
                matrix = torch.cat((matrix, embedding), dim=0)
            else:
                continue
        return matrix[1:, :]

    def initial_matrix(self):
        matrix = []
        for index in range(self.__len__()):
            matrix.append(self.sentence2matrix(self.sentences[index], OOV_strategy=self.OOV_strategy))

            if index / 1000 == index // 1000:
                print('Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                      (index / self.__len__() * 100, psutil.virtual_memory().percent, psutil.cpu_percent(1)))
        return matrix

    def __getitem__(self, index):
        input = self.matrix[index]

        if index < self.pos_no:
            ground_truth = torch.FloatTensor([1, 0])
        else:
            ground_truth = torch.FloatTensor([0, 1])

        return input, ground_truth

    def __len__(self):
        return len(self.sentences)
