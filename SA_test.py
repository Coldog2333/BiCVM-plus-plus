import sys
import getopt
import torch
import matplotlib.pyplot as plt
from network import SAnet, Net4SA
from load_data import CorpusLoader4SA

# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
# Program parameter
gpuID = 1
ACTIVATION = 'penalized tanh'
BIDIRECTION = False
full_model = 'temp'
TASK = ''

SAdir = '../data/aclImdb_v1/aclImdb/test'
en_word2vec = '../word2vec/en/enwiki_300.model'
oov_embedding_file = './models/en'

if sys.argv[1] in ['-h', '--help']:
    print("""BiCVM++ version beta
usage: python3 SA_test.py [[option] [value]]...
options:
--act          activation utilized in Pipeline
               valid values: [tanh, ptf, penalized tanh]. default: penalized tanh
--bid          Use bidirectional LSTM? [T/F]. default: False
--model        the name of model you want to use. default: temp 
--gpuID        the No. of the GPU you want to use. default: No.1
--task         special for the baseline Just for Sentiment Analysis.
               valid values: ['Just4SA', 'just', 'Just']
-h, --help     get help.""")
    exit(0)

# --------------------------------------------------------------
# Hyper Parameters
EPOCH = 25
LR = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
LR_strategy = []

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--act':                                    # activation
        if strArgument in ['tanh']:
            ACTIVATION = 'tanh'
        elif strArgument in ['penalized_tanh', 'ptf']:
            ACTIVATION = 'penalized tanh'
    elif strOption == '--bid':
        if strArgument in ['True', 'TRUE', 'true', 'T']:
            BIDIRECTION = True
        elif strArgument in ['False', 'FALSE', 'false', 'F']:
            BIDIRECTION = False
    elif strOption == '--model':
        full_model = './models/' + strArgument
    elif strOption == '--gpuID':                                # gpu id
        gpuID = int(strArgument)
        torch.cuda.set_device(gpuID)
    elif strOption == '--task':
        TASK = strArgument

Dataset = CorpusLoader4SA(SAdir=SAdir, mode='MemoryFriendly', word2vec=en_word2vec, cut=200, OOV_strategy='random', oov_embedding_file=oov_embedding_file)
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=1, shuffle=False)
sample_size = Dataset.__len__()


if TASK == '':
    net = SAnet(load_pretrain=False, activation=ACTIVATION, bidirection=BIDIRECTION, GPU_ID=gpuID)
elif TASK in ['Just4SA', 'just', 'Just']:
    net = Net4SA(activation=ACTIVATION)
else:
    raise NameError('Unknown [-task]')

net.load_state_dict(torch.load(full_model + '.pkl'))
net.cuda()
net.eval()

t = 0
count = 0

TP = 0
TN = 0

for step, (s1, ground_truth) in enumerate(train_loader):
    s1 = s1.cuda()
    ground_truth = ground_truth.cuda()

    label = net(s1)

    if (label[:, :, 0] >= label[:, :, 1] and ground_truth[:, 0] == 1):
        TP += 1
        t += 1
    elif (label[:, :, 0] < label[:, :, 1] and ground_truth[:, 1] == 1):
        TN += 1
        t += 1

    count += 1
    print('%d/%d' % (count, sample_size))

print('Accuracy: %f' % (t / count))
print('F1 Score: %f' % (2 * TP / (count + TP - TN)))