import numpy as np
import torch
import gensim
import matplotlib.pyplot as plt
from network import Net4Check, MultiCVM

GPU = 0
torch.cuda.set_device(GPU)
plt.switch_backend('agg')

full_model = './models/MultiCVM_170K_random_best_params.pkl'
# en_word2vec_model = '../word2vec/en/en.bin'
en_word2vec_model = '../word2vec/en/enwiki_300.model'
de_word2vec_model = '../word2vec/de/de.bin'

def sentence2matrix(sentence):
    matrix = torch.zeros(1, 300)
    for i in range(len(sentence)):
        embedding = word2vec_embedding(sentence[i])
        matrix = torch.cat((matrix, embedding), dim=0)
    return matrix[1:, :]

def word2vec_embedding(word):  # lang = 0/1
    return torch.FloatTensor(word2vec.wv.__getitem__(word)).view(1, 300)

def cosine_similarity(v1, v2):
    if 'torch' in str(type(v1)):
        v1 = v1.cpu().detach().numpy()
    if 'torch' in str(type(v2)):
        v2 = v2.cpu().detach().numpy()
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2))


net = MultiCVM(mode='hidden', GPU_ID=GPU)
net.load_state_dict(torch.load(full_model))
net.cuda()
net.eval()

word2vec = gensim.models.word2vec.Word2Vec.load(en_word2vec_model)

sentence1 = 'The pepper is very hot .'.split(' ')
sentence2 = 'This food is spicy .'.split(' ')
sentence3 = 'Today is cold .'.split(' ')


matrix1 = sentence2matrix(sentence1)
matrix1 = matrix1.view(1, matrix1.size(0), matrix1.size(1)).cuda()
matrix2 = sentence2matrix(sentence2)
matrix2 = matrix2.view(1, matrix2.size(0), matrix2.size(1)).cuda()
matrix3 = sentence2matrix(sentence3)
matrix3 = matrix3.view(1, matrix3.size(0), matrix3.size(1)).cuda()

embedding_hot = net(matrix1, 0, 0)[:, -1, :].view(300).cpu().detach().numpy()
embedding_spicy = net(matrix2, 0, 0)[:, -1, :].view(300).cpu().detach().numpy()
embedding_cold = net(matrix3, 0, 0)[:, -1, :].view(300).cpu().detach().numpy()

w2v_hot = word2vec.wv.__getitem__('hot')
w2v_spicy = word2vec.wv.__getitem__('spicy')
w2v_cold = word2vec.wv.__getitem__('cold')

print('期望的结果应该是: cvm中hot和spicy的相似度比w2v的大, 而hot和cold的相似度比w2v的小.')
similarity_hot_spicy_cvm = cosine_similarity(embedding_hot, embedding_spicy)
similarity_hot_cold_cvm = cosine_similarity(embedding_hot, embedding_cold)

similarity_hot_spicy_w2v = cosine_similarity(w2v_hot, w2v_spicy)
similarity_hot_cold_w2v = cosine_similarity(w2v_hot, w2v_cold)

print(similarity_hot_cold_cvm, similarity_hot_cold_w2v)
print(similarity_hot_spicy_cvm, similarity_hot_spicy_w2v)