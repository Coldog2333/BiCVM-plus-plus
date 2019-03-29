import os
import random
import datetime
import gensim
import psutil
import torch
import torch.utils.data

embedding_dim = 900


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, corpus1, corpus2, word2vec1='../word2vec/en/en.bin', word2vec2='../word2vec/de/de.bin'):
        self.sentences1 = self.load_sentences(corpus1)
        self.sentences2 = self.load_sentences(corpus2)

        self.word2vec_1 = gensim.models.word2vec.Word2Vec.load(word2vec1)
        self.word2vec_2 = gensim.models.word2vec.Word2Vec.load(word2vec2)

        self.oov_embedding_dict1 = dict()   # out-of-vocabulary词的词向量
        self.oov_embedding_dict2 = dict()   # out-of-vocabulary词的词向量

    def load_sentences(self, corpus):
        sentences = []
        with open(corpus, 'r', encoding='utf8') as f:
            raw = f.read().splitlines()
        for r in raw:
            sentences.append(r.split(' '))
        return sentences

    def word2vec_embedding(self, word, lang):
        if lang == 1:
            try:
                return torch.FloatTensor(self.word2vec_1.wv.__getitem__(word))
            except:
                if word not in self.oov_embedding_dict1.keys():
                    self.oov_embedding_dict1[word] = torch.rand((1, 300))
                return self.oov_embedding_dict1[word]
        elif lang == 2:
            try:
                return torch.FloatTensor(self.word2vec_2.wv.__getitem__(word))
            except:
                if word not in self.oov_embedding_dict2.keys():
                    self.oov_embedding_dict2[word] = torch.rand((1, 300))
                return self.oov_embedding_dict2[word]

    def sentence2matrix(self, sentence, lang, embed):
        if embed == 'onehot':
            matrix = torch.empty(len(sentence), embedding_dim)
            for i in range(len(sentence)):
                matrix[i, :] = self.onehot_embedding(sentence[i], lang=lang)
        elif embed == 'word2vec':
            matrix = torch.empty(len(sentence), 300)
            for i in range(len(sentence)):
                matrix[i, :] = self.word2vec_embedding(sentence[i], lang=lang)
        else:
            raise NameError('Invalid [-embed]')
        return matrix

    def __getitem__(self, index):
        input1 = self.sentence2matrix(self.sentences1[index], lang=1, embed='word2vec')
        input2 = self.sentence2matrix(self.sentences2[index], lang=2, embed='word2vec')

        disturb_input = []
        for i in range(3):
            r = random.randint(0, self.__len__() - 1)
            disturb_input.append(self.sentence2matrix(self.sentences2[r], lang=2, embed='word2vec'))

        return input1, input2, disturb_input

    def __len__(self):
        return len(self.sentences1)


class CorpusLoader(torch.utils.data.Dataset):
    def __init__(self, corpus1, corpus2, word2vec1='../word2vec/en/en.bin', word2vec2='../word2vec/de/de.bin'):
        self.sentences1 = self.load_sentences(corpus1)
        self.sentences2 = self.load_sentences(corpus2)

        self.word2vec_1 = gensim.models.word2vec.Word2Vec.load(word2vec1)
        self.word2vec_2 = gensim.models.word2vec.Word2Vec.load(word2vec2)

        self.oov_embedding_dict1 = dict()   # out-of-vocabulary词的词向量
        self.oov_embedding_dict2 = dict()   # out-of-vocabulary词的词向量

        print('Initializing sentence matrices...')
        self.matrix1 = self.initial_matrix(lang=1, embed='word2vec')
        self.matrix2 = self.initial_matrix(lang=2, embed='word2vec')
        print('Matrices are ready.')

    def load_sentences(self, corpus):
        sentences = []
        with open(corpus, 'r', encoding='utf8') as f:
            raw = f.read().splitlines()
        for r in raw:
            sentences.append(r.split(' '))
        return sentences

    def word2vec_embedding(self, word, lang):
        if lang == 1:
            try:
                return torch.FloatTensor(self.word2vec_1.wv.__getitem__(word))
            except:
                if word not in self.oov_embedding_dict1.keys():
                    self.oov_embedding_dict1[word] = torch.rand((1, 300))
                return self.oov_embedding_dict1[word]
        elif lang == 2:
            try:
                return torch.FloatTensor(self.word2vec_2.wv.__getitem__(word))
            except:
                if word not in self.oov_embedding_dict2.keys():
                    self.oov_embedding_dict2[word] = torch.rand((1, 300))
                return self.oov_embedding_dict2[word]

    def sentence2matrix(self, sentence, lang, embed):
        if embed == 'onehot':
            matrix = torch.empty(len(sentence), embedding_dim)
            for i in range(len(sentence)):
                matrix[i, :] = self.onehot_embedding(sentence[i], lang=lang)
        elif embed == 'word2vec':
            matrix = torch.empty(len(sentence), 300)
            for i in range(len(sentence)):
                matrix[i, :] = self.word2vec_embedding(sentence[i], lang=lang)
        else:
            raise NameError('Invalid [-embed]')
        return matrix

    def initial_matrix(self, lang, embed):
        matrix = []
        for index in range(self.__len__()):
            matrix.append(self.sentence2matrix(self.sentences1[index], lang=lang, embed=embed))

            if index / 1000 == index // 1000:
                print('Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                      (index / self.__len__() * 100, psutil.virtual_memory().percent, psutil.cpu_percent(1)))
        return matrix

    def out__getitem__(self, index):
        input1 = self.sentence2matrix(self.sentences1[index], lang=1, embed='word2vec')
        input2 = self.sentence2matrix(self.sentences2[index], lang=2, embed='word2vec')

        disturb_input = []
        for i in range(3):
            r = random.randint(0, self.__len__() - 1)
            disturb_input.append(self.sentence2matrix(self.sentences2[r], lang=2, embed='word2vec'))

        return input1, input2, disturb_input

    def __getitem__(self, index):
        input1 = self.matrix1[index]
        input2 = self.matrix2[index]

        disturb_input = []
        for i in range(3):
            r = random.randint(0, self.__len__() - 1)
            disturb_input.append(self.matrix2[r])

        return input1, input2, disturb_input

    def __len__(self):
        return len(self.sentences1)


class MemoryFriendlyLoader4SA(torch.utils.data.Dataset):
    def __init__(self, SAdir='../data/aclImdb_v1/aclImdb/train/', word2vec='../word2vec/en/en.bin'):
        self.sentences = []
        # pos
        for p in os.listdir(os.path.join(SAdir, 'pos')):
            with open(os.path.join(SAdir, 'pos', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:200])

        self.pos_no = len(self.sentences)       # 小于self.pos_no的都是pos
        # neg
        for p in os.listdir(os.path.join(SAdir, 'neg')):
            with open(os.path.join(SAdir, 'neg', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:200])

        self.word2vec = gensim.models.word2vec.Word2Vec.load(word2vec)
        self.oov_embedding_dict = dict()   # out-of-vocabulary词的词向量


    def word2vec_embedding(self, word):
        try:
            return torch.FloatTensor(self.word2vec.wv.__getitem__(word))
        except:
            if word not in self.oov_embedding_dict.keys():
                self.oov_embedding_dict[word] = torch.rand((1, 300))
            return self.oov_embedding_dict[word]

    def sentence2matrix(self, sentence):
        matrix = torch.empty(len(sentence), 300)
        for i in range(len(sentence)):
            matrix[i, :] = self.word2vec_embedding(sentence[i])
        return matrix

    def __getitem__(self, index):
        input1 = self.sentence2matrix(self.sentences[index])

        if index < self.pos_no:
            ground_truth = torch.FloatTensor([1, 0])
        else:
            ground_truth = torch.FloatTensor([0, 1])

        return input1, ground_truth

    def __len__(self):
        return len(self.sentences)


class CorpusLoader4SA(torch.utils.data.Dataset):
    def __init__(self, SAdir='../data/aclImdb_v1/aclImdb/train/', word2vec='../word2vec/en/en.bin'):
        self.sentences = []
        # pos
        for p in os.listdir(os.path.join(SAdir, 'pos')):
            with open(os.path.join(SAdir, 'pos', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:200])

        self.pos_no = len(self.sentences)       # 小于self.pos_no的都是pos
        # neg
        for p in os.listdir(os.path.join(SAdir, 'neg')):
            with open(os.path.join(SAdir, 'neg', p), 'r', encoding='utf8') as f:
                self.sentences.append(f.read().split(' ')[:200])

        self.word2vec = gensim.models.word2vec.Word2Vec.load(word2vec)
        self.oov_embedding_dict = dict()   # out-of-vocabulary词的词向量

        print('Initializing sentence matrices...')
        self.matrix = self.initial_matrix()
        print('Matrices are ready.')

    def word2vec_embedding(self, word):
        try:
            return torch.FloatTensor(self.word2vec.wv.__getitem__(word))
        except:
            if word not in self.oov_embedding_dict.keys():
                self.oov_embedding_dict[word] = torch.rand((1, 300))
            return self.oov_embedding_dict[word]

    def sentence2matrix(self, sentence):
        matrix = torch.empty(len(sentence), 300)
        for i in range(len(sentence)):
            matrix[i, :] = self.word2vec_embedding(sentence[i])
        return matrix

    def initial_matrix(self):
        matrix = []
        for index in range(self.__len__()):
            matrix.append(self.sentence2matrix(self.sentences[index]))

            if index / 1000 == index // 1000:
                print('Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                      (index / self.__len__() * 100, psutil.virtual_memory().percent, psutil.cpu_percent(1)))
        return matrix

    def out__getitem__(self, index):
        input1 = self.sentence2matrix(self.sentences[index])

        if index < self.pos_no:
            ground_truth = torch.FloatTensor([1, 0])
        else:
            ground_truth = torch.FloatTensor([0, 1])

        return input1, ground_truth

    def __getitem__(self, index):
        input1 = self.matrix[index]

        if index < self.pos_no:
            ground_truth = torch.FloatTensor([1, 0])
        else:
            ground_truth = torch.FloatTensor([0, 1])

        return input1, ground_truth

    def __len__(self):
        return len(self.sentences)