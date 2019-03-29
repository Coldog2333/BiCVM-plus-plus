import numpy as np
import torch
import gensim
import matplotlib.pyplot as plt
from network import Net4Check

torch.cuda.set_device(1)
plt.switch_backend('agg')

full_model = './models/first_final_params.pkl'
en_word2vec_model = '../word2vec/en/en.bin'
de_word2vec_model = '../word2vec/de/de.bin'

net = Net4Check()
net.load_state_dict(torch.load(full_model))
net.cuda()
net.eval()

check_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
check_de = ['Montag', 'Dienstag', 'Mittwoch','Donnerstag', 'Freitag', 'Samstag','Sonntag',
            'Januar', 'Februar', 'März', 'April', 'Kann', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']

word2vec_en = gensim.models.word2vec.Word2Vec.load(en_word2vec_model)
word2vec_de = gensim.models.word2vec.Word2Vec.load(de_word2vec_model)

word2vec_similarities = []
hidden_similarities = []
for i in range(len(check_en)):
    v_en = word2vec_en.wv.__getitem__(check_en[i])
    v_de = word2vec_de.wv.__getitem__(check_de[i])
    word2vec_similarity = np.dot(v_en, v_de) / np.sqrt(np.dot(v_en, v_en)) / np.sqrt(np.dot(v_de, v_de))
    word2vec_similarities.append(word2vec_similarity)

    h_en = net(torch.FloatTensor(v_en).view(1, 1, v_en.shape[0]).cuda())
    h_de = net(torch.FloatTensor(v_de).view(1, 1, v_de.shape[0]).cuda())
    hidden_similarity = torch.dot(h_en[0, 0, :], h_de[0, 0, :]) / torch.sqrt(torch.dot(h_en[0, 0, :], h_en[0, 0, :])) / torch.sqrt(torch.dot(h_de[0, 0, :], h_de[0, 0, :]))
    hidden_similarities.append(hidden_similarity.cpu().item())

for i in range(len(check_en)):
    print('[%f  %f]' % (word2vec_similarities[i], hidden_similarities[i]))