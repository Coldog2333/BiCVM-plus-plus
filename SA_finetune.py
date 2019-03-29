import os
import datetime
import psutil
import torch
import matplotlib.pyplot as plt
from network import SAnet
from load_data import MemoryFriendlyLoader4SA, CorpusLoader4SA

torch.cuda.set_device(1)
plt.switch_backend('agg')

SAdir = '../data/aclImdb_v1/aclImdb/train'
pretrain_MultiCVM = './models/full_MultiCVM_best_params.pkl'

# --------------------------------------------------------------
# Hyper Parameters
EPOCH = 20
WEIGHT_DECAY = 1 * 1e-4
BATCH_SIZE = 1
LR = 1e-4
LR_strategy = []
Training_pic_path = 'Training_result_SA1.jpg'
model_name = 'SA1'
model_information_txt = model_name + '_info.txt'

Dataset = CorpusLoader4SA(SAdir=SAdir, word2vec='../word2vec/en/en.bin')
# Dataset = MemoryFriendlyLoader4SA(SAdir=SAdir, word2vec='../word2vec/en/en.bin')
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
sample_size = Dataset.__len__()
# --------------------------------------------------------------
# some functions
def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s


def delta_time(datetime1, datetime2):
    if datetime1 > datetime2:
        datetime1, datetime2 = datetime2, datetime1
    second = 0
    # second += (datetime2.year - datetime1.year) * 365 * 24 * 3600
    # second += (datetime2.month - datetime1.month) * 30 * 24 * 3600
    second += (datetime2.day - datetime1.day) * 24 * 3600
    second += (datetime2.hour - datetime1.hour) * 3600
    second += (datetime2.minute - datetime1.minute) * 60
    second += (datetime2.second - datetime1.second)
    return second
# --------------------------------------------------------------
net = SAnet(pretrain_MultiCVM)
net.cuda()

MultiCVM_params = list(map(id, net.BiCVM.parameters()))
SAnet_params = filter(lambda p: id(p) not in MultiCVM_params, net.parameters())
# optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer = torch.optim.Adam([{'params': SAnet_params}, {'params': net.BiCVM.parameters(), 'lr': LR / 10}],
                             lr=LR,
                             weight_decay=WEIGHT_DECAY)
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
