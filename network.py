import torch


def init_linear_layer(linear_layer, initial_method='orthogonal'):
    if initial_method == 'orthogonal':
        torch.nn.init.orthogonal_(linear_layer.weight)
        try:
            torch.nn.init.orthogonal_(linear_layer.bias)
        except:
            print('It seems that there is no bias...')
    else:
        raise NameError('Unknown [-initial_method]')


def init_lstm_layer(lstm_layer, initial_method='orthogonal'):
    if initial_method == 'orthogonal':
        for weight in lstm_layer.parameters():
            torch.nn.init.orthogonal_(weight)
    else:
        raise NameError('Unknown [-initial_method]')


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, v1, v2, vrs):
        loss = 0
        loss0 = torch.sum(torch.pow(v1 - v2, 2))
        for vr in vrs:
            loss += 1 + loss0 - torch.sum(torch.pow(v1 - vr, 2))
        return torch.nn.functional.relu(loss)


class Penalized_tanh(torch.nn.Module):
    def __init__(self):
        super(Penalized_tanh, self).__init__()

    def forward(self, tensor):
        return 0.75 * torch.relu(torch.tanh(tensor)) + 0.25 * torch.tanh(tensor)


class CVM(torch.nn.Module):
    def __init__(self, lang, GPU_ID=0, activation='penalized tanh', bidirection=False):
        super(CVM, self).__init__()
        self.language = lang
        self.lstm1 = torch.nn.LSTM(input_size=300, hidden_size=300, batch_first=True, bidirectional=bidirection)   # 用于调整词向量(隐层activation)
        self.lstm2 = torch.nn.LSTM(input_size=300, hidden_size=512, batch_first=True, bidirectional=bidirection)   # 用于生成句向量
        self.attention = Attention(GPU_ID=GPU_ID, activation=activation)
        self.Wr = torch.nn.Linear(in_features=512, out_features=512,bias=False)
        self.Wp = torch.nn.Linear(in_features=512, out_features=512, bias=False)

        if activation == 'tanh':
            self.ptf = torch.tanh
        elif activation == 'penalized tanh':
            self.ptf = Penalized_tanh()
        else:
            raise NameError('Unknown parameters [-activation]')

        if False:
            init_lstm_layer(self.lstm1)
            init_lstm_layer(self.lstm2)
            init_linear_layer(self.Wr)
            init_linear_layer(self.Wp)

    def forward(self, sentence):
        out1, (h_n1, c_n1) = self.lstm1(sentence)               # out1实际上就属于contextualized word embedding
        out2, (h_n2, c_n2) = self.lstm2(out1)
        r = self.attention(out2, h_n2)
        h_star = self.ptf(self.Wr(r) + self.Wp(h_n2))
        return h_star, out1


class Attention(torch.nn.Module):
    def __init__(self, GPU_ID=0, activation='penalized tanh'):
        super(Attention, self).__init__()
        self.GPU_ID = GPU_ID
        self.Wh = torch.nn.Linear(in_features=512, out_features=512, bias=False)    # 即所谓trainable matrix
        self.WN = torch.nn.Linear(in_features=512, out_features=512, bias=False)
        self.w = torch.nn.Linear(in_features=512, out_features=1, bias=False)

        if activation == 'tanh':
            self.ptf = torch.tanh
        elif activation == 'penalized tanh':
            self.ptf = Penalized_tanh()
        else:
            raise NameError('Unknown parameters [-activation]')

        if False:
            init_linear_layer(self.Wh)
            init_linear_layer(self.WN)

    def forward(self, H, h_N):
        # H: [batch size, N, dim]
        # h_N: [batch size, 1, dim]
        eN = torch.ones((H.size(0), H.size(1), 1)).cuda(self.GPU_ID)   # eN: [batch size, 1, N]
        M = self.ptf(self.Wh(H) + self.WN(eN * h_N))
        alpha = torch.softmax(self.w(M), -1).permute(0, 2, 1)
        r = torch.matmul(alpha, H)                       # r: [batch_size, dim, 1]
        return r


class MultiCVM(torch.nn.Module):
    def __init__(self, mode='train', GPU_ID=0, activation='penalized tanh', freeze_flag=False, bidirection=False):
        super(MultiCVM, self).__init__()
        self.CVM1 = CVM(lang='English', GPU_ID=GPU_ID, activation=activation, bidirection=bidirection)
        self.CVM2 = CVM(lang='German', GPU_ID=GPU_ID, activation=activation, bidirection=bidirection)

        self.mode = mode

        if freeze_flag == True:
            for p in self.CVM1.parameters():
                p.requires_grad = False
            for p in self.CVM2.parameters():
                p.requires_grad = False

    def forward(self, s1, s2, s_r):
        if self.mode == 'train':
            # s1, s2是两句不同语言的平行译文, 分别表示为一个矩阵(详情见CVM)
            cvm1_out, _ = self.CVM1(s1)
            cvm2_out, _ = self.CVM2(s2)

            cvm2_out_sr, _ = self.CVM2(s_r[0].cuda())
            for i in range(1, len(s_r)):
                cvm2_out_sr = torch.cat((cvm2_out_sr, self.CVM2(s_r[i].cuda())[0]))

            return cvm1_out, cvm2_out, cvm2_out_sr

        elif 'eval' in self.mode:
            if self.mode[-1] == '1':
                cvm1_out, _ = self.CVM1(s1)
                return cvm1_out
            elif self.mode[-1] == '2':
                cvm2_out, _ = self.CVM2(s1)
                return cvm2_out
            else:
                raise NameError('Unknown cvm number [0/1]')

        elif self.mode == 'hidden':
            _, out = self.CVM1(s1)
            return out
        else:
            raise NameError('Unknown mode.')


# Standard Pipeline for Sentiment Analysis
class SAnet(torch.nn.Module):
    def __init__(self, MultiCVM_model='', load_pretrain=True, GPU_ID=0, activation='penalized tanh',
                 freeze_MultiCVM=False, bidirection=False):
        super(SAnet, self).__init__()

        self.BiCVM = MultiCVM(mode='eval1', GPU_ID=GPU_ID, activation=activation, bidirection=bidirection)
        if load_pretrain == True:
            self.BiCVM.load_state_dict(torch.load(MultiCVM_model))

        if freeze_MultiCVM == True:
            for p in self.BiCVM.parameters():
                p.requires_grad = False

        self.MLP1 = torch.nn.Linear(in_features=512, out_features=300)
        self.MLP2 = torch.nn.Linear(in_features=300, out_features=2)

        if activation == 'tanh':
            self.ptf = torch.tanh
        elif activation == 'penalized tanh':
            self.ptf = Penalized_tanh()
        else:
            raise NameError('Unknown parameters [-activation]')

    def forward(self, sentence):
        cvm1_out = self.BiCVM(sentence, 0, 0)
        out1 = self.MLP1(cvm1_out)
        out = self.ptf(out1)
        out2 = self.MLP2(out)
        label = torch.softmax(out2, -1)
        return label


# Just for Sentiment Analysis
class Net4SA(torch.nn.Module):
    def __init__(self, activation='penalized tanh'):
        super(Net4SA, self).__init__()
        self.MLP1 = torch.nn.Linear(in_features=300, out_features=300)
        self.MLP2 = torch.nn.Linear(in_features=300, out_features=2)

        if activation == 'tanh':
            self.ptf = torch.tanh
        elif activation == 'penalized tanh':
            self.ptf = Penalized_tanh()
        else:
            raise NameError('Unknown parameters [-activation]')


    def forward(self, sentence):
        out = sentence[:, 0, :]
        for i in range(1, sentence.size(1)):
            out += sentence[:, i, :]
        out = out.view(out.size(0), 1, out.size(1))

        mlp_out1 = self.MLP1(out)
        mlp_out = self.ptf(mlp_out1)
        mlp_out2 = self.MLP2(mlp_out)
        label = torch.softmax(mlp_out2, -1)
        return label


class ExtraFeature(torch.nn.Module):
    def __init__(self, activation='penalized tanh', load_pretrain=False, GPU_ID=0, bidirection=False):
        super(ExtraFeature, self).__init__()
        self.bicvm_exist = load_pretrain
        if load_pretrain == False:
            self.MLP1 = torch.nn.Linear(in_features=300, out_features=300)
            self.MLP2 = torch.nn.Linear(in_features=300, out_features=2)
        else:
            self.MLP1 = torch.nn.Linear(in_features=300 + 512, out_features=300)
            self.MLP2 = torch.nn.Linear(in_features=300, out_features=2)
            self.BiCVM = MultiCVM(mode='eval1', GPU_ID=GPU_ID, activation=activation, bidirection=bidirection)

        if activation == 'tanh':
            self.ptf = torch.tanh
        elif activation == 'penalized tanh':
            self.ptf = Penalized_tanh()
        else:
            raise NameError('Unknown parameters [-activation]')


    def forward(self, sentence):
        out = sentence[:, 0, :]
        for i in range(1, sentence.size(1)):
            out += sentence[:, i, :]
        out = out.view(out.size(0), 1, out.size(1))

        if self.bicvm_exist == True:
            bicvm = self.BiCVM(sentence, 0, 0)
            print(bicvm.shape, out.shape)
            out = torch.cat((out, bicvm), dim=-1)

        mlp_out1 = self.MLP1(out)
        mlp_out = self.ptf(mlp_out1)
        mlp_out2 = self.MLP2(mlp_out)
        label = torch.softmax(mlp_out2, -1)
        return label


# Check word embedding in CVM.
class Net4Check(torch.nn.Module):
    def __init__(self, GPU_ID=0):
        super(Net4Check, self).__init__()
        self.CVM1 = CVM(lang='English', GPU_ID=GPU_ID)
        self.CVM2 = CVM(lang='German', GPU_ID=GPU_ID)

    def forward(self, s1):
        cvm1_out, hidden = self.CVM1(s1)
        return cvm1_out



#####################################################
# These codes are abandoned.
#####################################################
# class Net4Ablation(torch.nn.Module):
#     def __init__(self):
#         super(Net4Ablation, self).__init__()
#         self.lstm1 = torch.nn.LSTM(input_size=300, hidden_size=300, batch_first=True)   # 用于调整词向量(隐层activation)
#         self.lstm2 = torch.nn.LSTM(input_size=300, hidden_size=512, batch_first=True)   # 用于生成句向量
#
#         self.MLP1 = torch.nn.Linear(in_features=512, out_features=300)
#         self.MLP2 = torch.nn.Linear(in_features=300, out_features=2)
#
#         self.ptf = Penalized_tanh()
#
#     def forward(self, sentence):
#         out1, _ = self.lstm1(sentence)
#         _, (h_n2, _) = self.lstm2(out1)
#
#         mlp_out1 = self.MLP1(h_n2)
#         # mlp_out = torch.tanh(mlp_out1)
#         mlp_out = self.ptf(mlp_out1)
#         mlp_out2 = self.MLP2(mlp_out)
#
#         label = torch.softmax(mlp_out2, -1)
#         return label