import torch

class CVM(torch.nn.Module):
    def __init__(self, lang):
        super(CVM, self).__init__()
        self.language = lang
        self.lstm1 = torch.nn.LSTM(input_size=300, hidden_size=300, batch_first=True)   # 用于调整词向量(隐层activation)
        self.lstm2 = torch.nn.LSTM(input_size=300, hidden_size=512, batch_first=True)   # 用于生成句向量
        # self.Wh = torch.nn.Linear(in_features=0, out_features=0, bias=False)          # 即所谓trainable matrix

    def forward(self, sentence):
        out1, (h_n1, c_n1) = self.lstm1(sentence)       # out1实际上就属于contextualized word embedding
        out2, (h_n2, c_n2) = self.lstm2(out1)
        return h_n2, out1


class MultiCVM(torch.nn.Module):
    def __init__(self, mode='train'):
        super(MultiCVM, self).__init__()
        # self.CVM3 = CVM(lang='Chinese')
        self.CVM1 = CVM(lang='English')
        self.CVM2 = CVM(lang='German')

        self.mode = mode

    def forward(self, s1, s2, s_r):
        if self.mode == 'train':
            # s1, s2是两句不同语言的平行译文, 分别表示为一个矩阵(详情见CVM)
            cvm1_out, _ = self.CVM1(s1)
            cvm2_out, _ = self.CVM2(s2)

            cvm2_out_1, _ = self.CVM2(s_r[0].cuda())
            cvm2_out_2, _ = self.CVM2(s_r[1].cuda())
            cvm2_out_3, _ = self.CVM2(s_r[2].cuda())

            cvm2_out_sr = torch.cat((cvm2_out_1, cvm2_out_2, cvm2_out_3))

            return cvm1_out, cvm2_out, cvm2_out_sr
        elif self.mode == 'eval':
            cvm1_out, _ = self.CVM1(s1)
            return cvm1_out
        else:
            raise NameError('Unknown mode.')


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


class SAnet(torch.nn.Module):
    def __init__(self, MultiCVM_model, freeze_MultiCVM=False):
        super(SAnet, self).__init__()

        self.BiCVM = MultiCVM(mode='eval')
        self.BiCVM.load_state_dict(torch.load(MultiCVM_model))

        if freeze_MultiCVM == True:
            for p in self.BiCVM.parameters():
                p.requires_grad = False

        self.MLP1 = torch.nn.Linear(in_features=512, out_features=300)
        self.MLP2 = torch.nn.Linear(in_features=300, out_features=2)

    def forward(self, sentence):
        cvm1_out = self.BiCVM(sentence, 0, 0)
        out1 = self.MLP1(cvm1_out)
        out = torch.tanh(out1)
        out2 = self.MLP2(out)
        label = torch.softmax(out2, -1)
        return label


class Net4Ablation(torch.nn.Module):
    def __init__(self):
        super(Net4Ablation, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=300, hidden_size=300, batch_first=True)   # 用于调整词向量(隐层activation)
        self.lstm2 = torch.nn.LSTM(input_size=300, hidden_size=512, batch_first=True)   # 用于生成句向量
        self.linear = torch.nn.Linear(in_features=512, out_features=2)

    def forward(self, sentence):
        out1, _ = self.lstm1(sentence)
        _, (h_n2, _) = self.lstm2(out1)
        out = self.linear(h_n2)
        label = torch.softmax(out, -1)
        return label


class Net4Check(torch.nn.Module):
    def __init__(self):
        super(Net4Check, self).__init__()
        self.CVM1 = CVM(lang='English')
        self.CVM2 = CVM(lang='German')

    def forward(self, s1):
        cvm1_out, hidden = self.CVM1(s1)
        return cvm1_out