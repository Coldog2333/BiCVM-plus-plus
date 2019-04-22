import os
import datetime
import psutil
import torch
import matplotlib.pyplot as plt
from utils import show_time, delta_time
from network import SAnet, Net4SA, ExtraFeature
from load_data import CorpusLoader4SA
import sys
import getopt

# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
# Program parameter
gpuID = 1
MODE = 'MemoryFriendly'
ACTIVATION = 'penalized tanh'
FREEZE = False
PRETRAIN = True
BIDIRECTION = False
TASK = ''

model_dir = '../v0.9.1/models'
# SAdir = '../data/aclImdb_v1/tiny/train'
SAdir = '../data/aclImdb_v1/aclImdb/train'
# pretrain_MultiCVM = './models/170K_3_best_params.pkl'
pretrain_MultiCVM = os.path.join(model_dir, 'BiCVMpp_170K_final_params.pkl')
en_word2vec = '../word2vec/en/enwiki_300.model'
# oov_embedding_file = './models/en'
oov_embedding_file = os.path.join(model_dir, 'en')

if sys.argv[1] in ['-h', '--help']:
    print("""BiCVM++ version beta
usage: python3 SA_finetune.py [[option] [value]]...
options:
--model        the name of model you want to save as.
--act          activation utilized in Pipeline
               valid values:[tanh, ptf, penalized tanh]. default: penalized tanh
--freeze       Do you want to freeze the parameters of CVM? [T/F]. default: False
--pretrain     Do you want to load pretrain weights? [T/F]. default: True
--bid          Use bidirectional LSTM? [T/F]. default: False
--mode         Do you want to use MemoryFriendlyLoader or CorpusLoader? [MemoryFriendly/Effective]. default: MemoryFriendly
--gpuID        the No. of the GPU you want to use. default: No.1
--task         convenient way to define a type of network.
               valid values: ['Just4SA', 'just', 'Just',
                              'full',
                              'joint',
                              'fixed',
                              'randomized',
                              'ExtraFeature', 'extra', 'Extra', 'ex', 'Ex']
-h, --help     get help.""")
    exit(0)

# --------------------------------------------------------------
# Hyper Parameters
EPOCH = 25
LR = 2 * 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1
LR_strategy = []

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [strParameter[2:] + '=' for strParameter in sys.argv[1::2]])[0]:
    if strOption == '--act':                                    # activation
        if strArgument in ['tanh']:
            ACTIVATION = 'tanh'
        elif strArgument in ['penalized_tanh', 'ptf']:
            ACTIVATION = 'penalized tanh'
    elif strOption == '--freeze':
        if strArgument in ['True', 'TRUE', 'true', 'T']:
            FREEZE = True
        elif strArgument in ['False', 'FALSE', 'false', 'F']:
            FREEZE = False
    elif strOption == '--pretrain':
        if strArgument in ['True', 'TRUE', 'true', 'T']:
            PRETRAIN = True
        elif strArgument in ['False', 'FALSE', 'false', 'F']:
            PRETRAIN = False
    elif strOption == '--bid':
        if strArgument in ['True', 'TRUE', 'true', 'T']:
            BIDIRECTION = True
        elif strArgument in ['False', 'FALSE', 'false', 'F']:
            BIDIRECTION = False
    elif strOption == '--model':
        model_name = strArgument
        model_information_txt = model_name + '_info.txt'
        Training_pic_path = model_name + '.jpg'
    elif strOption == '--gpuID':                                # gpu id
        gpuID = int(strArgument)
        torch.cuda.set_device(gpuID)
    elif strOption == '--task':
        TASK = strArgument
        if TASK == 'full':
            FREEZE = False
            PRETRAIN = True
        elif TASK == 'joint':
            FREEZE = False
            PRETRAIN = False
        elif TASK == 'fixed':
            FREEZE = True
            PRETRAIN = True
        elif TASK == 'randomized':
            FREEZE = True
            PRETRAIN = False
        elif TASK in ['Just4SA', 'just', 'Just']:
            pass
        elif TASK in ['ExtraFeature', 'extra', 'Extra', 'ex', 'Ex']:
            pass
        else:
            raise NameError('Unknown [-task]')
# --------------------------------------------------------------
# load data
Dataset = CorpusLoader4SA(SAdir=SAdir, mode=MODE, word2vec=en_word2vec, cut=200, OOV_strategy='random', oov_embedding_file=oov_embedding_file)
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True)
sample_size = Dataset.__len__()
# --------------------------------------------------------------
if TASK in ['Just4SA', 'just', 'Just']:
    net = Net4SA(activation=ACTIVATION)
elif TASK in ['ExtraFeature', 'extra', 'Extra', 'ex', 'Ex']:
    net = ExtraFeature(activation=ACTIVATION, load_pretrain=True, GPU_ID=gpuID, bidirection=BIDIRECTION)
else:
    net = SAnet(pretrain_MultiCVM, load_pretrain=PRETRAIN, freeze_MultiCVM=FREEZE, activation=ACTIVATION, GPU_ID=gpuID, bidirection=BIDIRECTION)
net.cuda()

# MultiCVM_params = list(map(id, net.BiCVM.parameters()))
# SAnet_params = filter(lambda p: id(p) not in MultiCVM_params, net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
# optimizer = torch.optim.Adam([{'params': SAnet_params}, {'params': net.BiCVM.parameters(), 'lr': LR / 10}],
#                              lr=LR,
#                              weight_decay=WEIGHT_DECAY)
loss_func = torch.nn.MSELoss()

# Training
prev_time = datetime.datetime.now()  # current time
print('%s  Start training...' % show_time(prev_time))
plotx = []
ploty = []
check_loss = 1

for epoch in range(EPOCH):
    losses = 0
    count = 0
    for step, (s1, ground_truth) in enumerate(train_loader):
        s1 = s1.cuda()
        ground_truth = ground_truth.cuda()

        label = net(s1)

        loss = loss_func(label, ground_truth)
        losses += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        count += len(s1)
        if count / 1000 == count // 1000:
            print('%s  Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                  (show_time(datetime.datetime.now()), count / sample_size * 100, psutil.virtual_memory().percent,
                   psutil.cpu_percent(1)))

    print('[%f, %f]' % (label[:, :, 0].cpu().item(), label[:, :, 1].cpu().item()))
    print('\n%s  epoch %d: Average_loss=%f\n' % (show_time(datetime.datetime.now()), epoch + 1, losses / (step + 1)))

    plotx.append(epoch + 1)
    ploty.append(losses / (step + 1))
    if epoch // 1 == epoch / 1:
        plt.plot(plotx, ploty)
        plt.savefig(Training_pic_path)

    if check_loss > losses / (step + 1):
        print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
        if not os.path.exists(os.path.join('.', 'models')):
            os.mkdir(os.path.join('.', 'models'))
        torch.save(net.state_dict(), os.path.join('.', 'models', model_name + '_best_params.pkl'))
        print('Saved.\n')
        check_point = losses / (step + 1)

net.cpu()
# Dataset.save_oov_embedding(oov_embedding_file)
plt.plot(plotx, ploty)
plt.savefig(Training_pic_path)

cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Training costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))

print('\n%s Saving model...' % show_time(datetime.datetime.now()))
if not os.path.exists(os.path.join('.', 'models')):
    os.mkdir(os.path.join('.', 'models'))

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
fp.write('\nTrain on %dK_%s\n' % (int(sample_size / 1000), 'IMDb'))
print("Training costs %02d:%02d:%02d" % (h, m, s), file=fp)
fp.close()


cur_time = datetime.datetime.now()
h, remainder = divmod(delta_time(prev_time, cur_time), 3600)
m, s = divmod(remainder, 60)
print('%s  Totally costs %02d:%02d:%02d' % (show_time(datetime.datetime.now()), h, m, s))
print('%s  All done.' % show_time(datetime.datetime.now()))
