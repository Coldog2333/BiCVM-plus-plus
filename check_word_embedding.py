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

fp = open('result.csv', 'w', encoding='utf8')

def cosine_similarity(v1, v2):
    if 'torch' in str(type(v1)):
        v1 = v1.cpu().detach().numpy()
    if 'torch' in str(type(v2)):
        v2 = v2.cpu().detach().numpy()
    return np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2))


# net = Net4Check()
net = MultiCVM(mode='eval', GPU_ID=GPU)
net.load_state_dict(torch.load(full_model))
net.cuda()
net.eval()

check_en = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December']
check_de = ['Januar', 'Februar', 'März', 'April', 'Kann', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November',
           'Dezember']

# check_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# check_de = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

word2vec_en = gensim.models.word2vec.Word2Vec.load(en_word2vec_model)
word2vec_de = gensim.models.word2vec.Word2Vec.load(de_word2vec_model)

fp.write('\n看一下用新的embedding, 不同语言的类似词之间会不会比原来的那个更类似\n')
word2vec_similarities = []
hidden_similarities = []
for i in range(len(check_en)):
    v_en = word2vec_en.wv.__getitem__(check_en[i])
    v_de = word2vec_de.wv.__getitem__(check_de[i])
    word2vec_similarity = np.dot(v_en, v_de) / np.sqrt(np.dot(v_en, v_en)) / np.sqrt(np.dot(v_de, v_de))
    word2vec_similarities.append(word2vec_similarity)

    h_en = net(torch.FloatTensor(v_en).view(1, 1, v_en.shape[0]).cuda(), 0, 0)
    h_de = net(torch.FloatTensor(v_de).view(1, 1, v_de.shape[0]).cuda(), 0, 0)
    hidden_similarity = cosine_similarity(h_en[0, 0, :], h_de[0, 0, :])
    hidden_similarities.append(hidden_similarity)

fp.write(' ,Word2Vec,MultiCVM,Delta\n')
for i in range(len(check_en)):
    # print('[%.3f  %.3f]' % (word2vec_similarities[i], hidden_similarities[i]))
    fp.write('%s,%.3f,%.3f,%.3f\n' % (check_en[i] + '-' + check_de[i], word2vec_similarities[i], hidden_similarities[i],
              hidden_similarities[i] - word2vec_similarities[i]))

fp.write('\n看一下用新的embedding, 同种语言的类似词之间的相似度会提高多少(en)\n')
fp.write(' \t')
for k in range(len(check_en)):
    fp.write(check_en[k] + ',')
fp.write('Aver.(%)\n')

total_error = 0
for k in range(len(check_en)):
    Old_similarities = []
    New_similarities = []
    Monday_old = word2vec_en.wv.__getitem__(check_en[k])
    Monday_new = net(torch.FloatTensor(Monday_old).view(1, 1, Monday_old.shape[0]).cuda(), 0, 0).cpu().detach().numpy()
    for i in range(len(check_en)):
        en_old = word2vec_en.wv.__getitem__(check_en[i])
        en_new = net(torch.FloatTensor(en_old).view(1, 1, en_old.shape[0]).cuda(), 0, 0).cpu().detach().numpy()

        Old_similarity = cosine_similarity(Monday_old, en_old)
        Old_similarities.append(Old_similarity)

        New_similarity = cosine_similarity(Monday_new[0, 0, :], en_new[0, 0, :])
        New_similarities.append(New_similarity)

    error = 0
    fp.write(check_en[k] + ',')
    for i in range(len(check_en)):
        # print('[%f  %f]' % (Old_similarities[i], New_similarities[i]))
        error += New_similarities[i] - Old_similarities[i]
        fp.write('%.3f,' % (New_similarities[i] - Old_similarities[i]))

    fp.write('%.3f%%\n' % (error / len(check_en) * 100))
    print('相似度平均提高:%f%%' % (error / len(check_en) * 100))
    total_error += error / len(check_en)

print('相似度总平均提高:%f%%' % (total_error / len(check_en) * 100))


fp.write('\n看一下新的word2vec下, 反义词之间会不会比原来的那个更不类似\n')
fp.write(' ,Word2Vec,MultiCVM,Delta,perc.\n')

check_positive = ['good', 'happy', 'big', 'polite', 'cold', 'fast', 'below', 'buy']
check_synonym = ['perfect', 'pleased', 'huge', 'courteous', 'chilly', 'quick', 'beneath', 'purchase']
check_negative = ['bad', 'sad', 'small', 'rude', 'hot', 'slow', 'above', 'sell']
total_error = 0
for i in range(len(check_positive)):
    en_pos = word2vec_en.wv.__getitem__(check_positive[i])
    en_neg = word2vec_en.wv.__getitem__(check_negative[i])

    h_pos = net(torch.FloatTensor(en_pos).view(1, 1, en_pos.shape[0]).cuda(), 0, 0).cpu().detach().numpy()
    h_neg = net(torch.FloatTensor(en_neg).view(1, 1, en_neg.shape[0]).cuda(), 0, 0).cpu().detach().numpy()

    error = cosine_similarity(h_pos[0, 0, :], h_neg[0, 0, :]) - cosine_similarity(en_pos, en_neg)

    fp.write(check_positive[i] + '-' + check_negative[i] + ',')
    fp.write('%.3f,%.3f,%.3f,%.3f%%\n' % (cosine_similarity(en_pos, en_neg), cosine_similarity(h_pos[0, 0, :], h_neg[0, 0, :]),
                                        error, error / cosine_similarity(en_pos, en_neg) * 100))
    print('相似度平均上升:%f%%' % (error / len(check_positive) * 100))
    total_error += error / len(check_positive)

print('相似度总平均上升:%f%%' % (total_error / len(check_positive) * 100))

total_error = 0
for i in range(len(check_positive)):
    en_pos = word2vec_en.wv.__getitem__(check_positive[i])
    en_syn = word2vec_en.wv.__getitem__(check_synonym[i])

    h_pos = net(torch.FloatTensor(en_pos).view(1, 1, en_pos.shape[0]).cuda(), 0, 0).cpu().detach().numpy()
    h_syn = net(torch.FloatTensor(en_syn).view(1, 1, en_neg.shape[0]).cuda(), 0, 0).cpu().detach().numpy()

    error = cosine_similarity(h_pos[0, 0, :], h_syn[0, 0, :]) - cosine_similarity(en_pos, en_syn)

    fp.write(check_positive[i] + '-' + check_synonym[i] + ',')
    fp.write('%.3f,%.3f,%.3f,%.3f%%\n' % (cosine_similarity(en_pos, en_syn), cosine_similarity(h_pos[0, 0, :], h_syn[0, 0, :]),
                                        error, error / cosine_similarity(en_pos, en_syn) * 100))
    print('相似度平均上升:%f%%' % (error / len(check_positive) * 100))
    total_error += error / len(check_positive)

print('相似度总平均上升:%f%%' % (total_error / len(check_positive) * 100))

fp.write('\n原来在word2vec的相似度排序 vs 现在在MultiCVM的相似度排序\n')
for i in range(len(check_positive)):
    fp.write(check_positive[i] + '|' + check_negative[i] + '|' + check_synonym[i] + ',')

    en_pos = word2vec_en.wv.__getitem__(check_positive[i])
    en_neg = word2vec_en.wv.__getitem__(check_negative[i])
    en_syn = word2vec_en.wv.__getitem__(check_synonym[i])

    h_pos = net(torch.FloatTensor(en_pos).view(1, 1, en_pos.shape[0]).cuda(), 0, 0).cpu().detach().numpy()
    h_neg = net(torch.FloatTensor(en_neg).view(1, 1, en_neg.shape[0]).cuda(), 0, 0).cpu().detach().numpy()
    h_syn = net(torch.FloatTensor(en_syn).view(1, 1, en_neg.shape[0]).cuda(), 0, 0).cpu().detach().numpy()

    if cosine_similarity(en_pos, en_neg) > cosine_similarity(en_pos, en_syn):
        fp.write(check_negative[i] + '>' + check_synonym[i])
    elif cosine_similarity(en_pos, en_neg) < cosine_similarity(en_pos, en_syn):
        fp.write(check_synonym[i] + '>' + check_negative[i])
    else:
        fp.write(check_synonym[i] + '=' + check_negative[i])

    fp.write(',')

    if cosine_similarity(h_pos[0, 0, :], h_neg[0, 0, :]) > cosine_similarity(h_pos[0, 0, :], h_syn[0, 0, :]):
        fp.write(check_negative[i] + '>' + check_synonym[i])
    elif cosine_similarity(h_pos[0, 0, :], h_neg[0, 0, :]) < cosine_similarity(h_pos[0, 0, :], h_syn[0, 0, :]):
        fp.write(check_synonym[i] + '>' + check_negative[i])
    else:
        fp.write(check_synonym[i] + '=' + check_negative[i])

    fp.write('\n')

fp.close()
