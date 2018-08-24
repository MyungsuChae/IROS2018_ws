import torch, torchvision
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from loss import get_loss, JointLoss, StaticLoss
import numpy as np

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for p in m.named_parameters():
                    if 'weight' in p[0]:
                        nn.init.xavier_normal_(p[1]).cuda()

class MultiNet(MyModule):
    def __init__(self, lstm_h=256, lstm_l=2, n_out=[4, 2], modal='multi', loss=None, stl_dim=-1):
        super().__init__()
        self.modal = modal
        self.loss = loss

        if modal=='audio':
            self.audionet = AudioNet()
            self.lstm = nn.LSTM(1280, lstm_h, lstm_l)
        elif modal=='video':
            self.videonet = VideoNet()
            self.lstm = nn.LSTM(2048, lstm_h, lstm_l)
        elif modal=='multi':
            self.audionet = AudioNet()
            self.videonet = VideoNet()
            self.lstm = nn.LSTM(3328, lstm_h, lstm_l)
        else:
            raise ValueError('Modal shoud be specified.')

        self.fc_emo = nn.Linear(lstm_h, n_out[0])
        self.fc_sex = nn.Linear(lstm_h, n_out[1])

        self.drop = nn.Dropout()
        self.loss = get_loss(loss=loss, n_out=len(n_out), stl_dim=stl_dim)
        self.init_weights()

    def forward(self, x_v, x_a, lengths):
        if self.modal=='audio':
            f = self.audionet(x_a)
        elif self.modal=='video':
            f = self.videonet(x_v)
        elif self.modal=='multi':
            f_v = self.videonet(x_v)
            f_a = self.audionet(x_a)
            f = torch.cat((f_a, f_v), 2)

        f = pack_padded_sequence(f, lengths, batch_first=True)
        f, _ = self.lstm(f)
        f, _ = pad_packed_sequence(f, batch_first=True)

        f = self.drop(f)
        f = f[np.arange(x_v.shape[0]), lengths-1, :]
        out_emo = self.fc_emo(f)
        out_sex = self.fc_sex(f)

        return out_emo, out_sex

    def calc_loss(self, pred, y):
        return self.loss(pred, y)

class AudioNet(MyModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 40, kernel_size=80)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(40, 40, kernel_size=4000)
        self.pool2 = nn.MaxPool1d(10)
        self.init_weights()

    def forward(self, x_a):
        n, c, l = x_a.shape
        f_a = self.conv1(F.pad(x_a,(40,39)))
        f_a = self.pool1(f_a)
        f_a = self.conv2(F.pad(f_a,(2000,1999)))
        f_a = f_a.permute(0,2,1).contiguous()
        f_a = self.pool2(f_a)
        f_a = f_a.view(n, -1, 1280)
        return f_a

class VideoNet(MyModule):
    def __init__(self):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.init_weights()

    def forward(self, x):
        b, cl, h, w = x.shape
        l = cl//3
        x = x.reshape(b*l, cl//l, h, w)
        f = self.feature_extractor(x)
        f = self.avgpool(f)
        f = f.reshape(b, l, 2048)
        return f
