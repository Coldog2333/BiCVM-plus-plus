import os
import datetime
import psutil
import torch
import matplotlib.pyplot as plt
from utils import show_time, delta_time
from network import MultiCVM, MyLoss
from load_data import CorpusLoader
import sys
import getopt

# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
# Program parameter
gpuID = 0
MODE = 'MemoryFriendly'
ACTIVATION = 'penalized tanh'
NOISE = 25
BIDIRECTION = False
model_name = ''
Dataset = ''

en_word2vec = '../word2vec/en/enwiki_300.model'
oov_embedding_files = ['./models/en', './models/de']

if sys.argv[1] in ['-h', '--help']:
    print("""BiCVM++ version beta
usage: python3 train.py [[option] [value]]...
options:
--data         Which dataset do you want to train on? 
               valid values: [tiny, 1K, 17K, 170K, full], default: tiny
--act          activation utilized in Pipeline
               valid values: [tanh, ptf, penalized tanh]. default: penalized tanh
--noise        The number of noisy samples. default: 25
--bid          Use bidirectional LSTM? [T/F]. default: False
--model        The name of model you want to save as. default value depends on what dataset you use.
--gpuID        The No. of the GPU you want to use. default: No.1
--mode         Do you want to use MemoryFriendlyLoader or CorpusLoader? 
               valid values: [MemoryFriendly/Effective], default: MemoryFriendly
-h, --help     get help.""")
    exit(0)

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--act':                                    # activation
        if strArgument in ['tanh']:
            ACTIVATION = 'tanh'
        elif strArgument in ['penalized_tanh', 'ptf']:
            ACTIVATION = 'penalized tanh'
    if strOption == '--data':
        Dataset = strArgument
        corpus1_file = '../data/training-parallel/standard/europarl-%s.de-en.en' % strArgument
        corpus2_file = '../data/training-parallel/standard/europarl-%s.de-en.de' % strArgument
    elif strOption == '--bid':
        if strArgument in ['True', 'TRUE', 'true', 'T']:
            BIDIRECTION = True
        elif strArgument in ['False', 'FALSE', 'false', 'F']:
            BIDIRECTION = False
    elif strOption == '--model':
        model_name = strArgument
    elif strOption == '--gpuID':                                # gpu id
        gpuID = int(strArgument)
        torch.cuda.set_device(gpuID)
    elif strOption == '--mode':
        MODE = strArgument

# --------------------------------------------------------------
# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 1
LR = 1e-5
LR_strategy = []
WEIGHT_DECAY = 1e-4

use_checkpoint = False
checkpoint_path = './checkpoints/checkpoint_0epoch.ckpt'
Training_pic_path = 'Training_result.jpg'

if model_name == '':
    model_name = 'BiCVMpp_' + Dataset

model_information_txt = model_name + '_info.txt'
OOV_strategy = 'random'

Dataset = CorpusLoader(corpus0=corpus1_file, corpus1=corpus2_file, mode=MODE,
                               num_of_noise=NOISE, word2vec0=en_word2vec, OOV_strategy=OOV_strategy)      # 如果内存允许
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True)
sample_size = Dataset.__len__()
# --------------------------------------------------------------
# some functions
def save_checkpoint(net, optimizer, epoch, losses, savepath):
    save_json = {
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)


def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses

# --------------------------------------------------------------
net = MultiCVM(GPU_ID=gpuID, activation=ACTIVATION)
net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_func = MyLoss()

# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []
start_epoch = 0
check_loss = 1

if use_checkpoint:
    net, optimizer, start_epoch, ploty = load_checkpoint(net, optimizer, checkpoint_path)
    plotx = list(range(len(ploty)))
    check_loss = min(ploty)

for epoch in range(EPOCH):
    losses = 0
    count = 0
    for step, (s1, s2, s_r) in enumerate(train_loader):
        s1 = s1.cuda()
        s2 = s2.cuda()

        v1, v2, vrs = net(s1, s2, s_r)

        loss = loss_func(v1, v2, vrs)
        losses += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        count += len(s1)
        if count / 1000 == count // 1000:
            print('%s  Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                  (show_time(datetime.datetime.now()), count / sample_size * 100, psutil.virtual_memory().percent,
                   psutil.cpu_percent(1)))

    print('\n%s  epoch %d: Average_loss=%f\n' % (show_time(datetime.datetime.now()), epoch + 1, losses / (step + 1)))

    plotx.append(epoch + 1)
    ploty.append(losses / (step + 1))
    if epoch // 1 == epoch / 1:
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)

    # checkpoint and then prepare for the next epoch
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    save_checkpoint(net, optimizer, epoch + 1, ploty, './checkpoints/checkpoint_%depoch.ckpt' % (epoch + 1))

    if check_loss > losses / (step + 1):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join('.', 'models')):
            os.mkdir(os.path.join('.', 'models'))
        torch.save(net.state_dict(), os.path.join('.', 'models', model_name + '_best_params.pkl'))
        print('Saved.\n')
        check_point = losses / (step + 1)

net.cpu()
Dataset.save_oov_embedding(oov_embedding_files)
plt.plot(plotx, ploty)
plt.savefig(Training_pic_path)

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))

print('\n%s Saving model...' % show_time(datetime.datetime.now()))
if not os.path.exists(os.path.join('.', 'models')):
    os.mkdir(os.path.join('.', 'models'))
# just save the parameters.
torch.save(net.state_dict(), os.path.join('.', 'models', model_name + '_final_params.pkl'))

print('\n%s  Collecting some information...' % show_time(datetime.datetime.now()))
fp = open(os.path.join('.', 'models', model_information_txt), 'w')
fp.write('Model Path:%s\n' % os.path.join('.', 'models', model_name + '_final_params.pkl'))
fp.write('\nModel Structure:\n')
print(net, file=fp)
fp.write('\nModel Hyper Parameters:\n')
fp.write('\tEpoch = %d\n' % EPOCH)
fp.write('\tBatch size = %d\n' % BATCH_SIZE)
fp.write('\tLearning rate = %f\n' % LR)
fp.write('\tWeight decay = %f\n' % WEIGHT_DECAY)
print('\tLR strategy = %s' % str(LR_strategy), file=fp)
fp.write('Loss: ')
for i in range(len(ploty)):
    fp.write('%f\t' % ploty[i])
fp.write('\nTrain on %dK_%s\n' % (int(sample_size / 1000), 'Europarl'))
print("Training costs %02d:%02d:%02d" % (h, m, s), file=fp)
fp.close()

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Totally costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))
print('%s  All done.' % show_time(datetime.datetime.now()))
