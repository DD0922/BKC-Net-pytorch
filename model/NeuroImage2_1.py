import torch.nn as nn
import torch.nn.functional as F
import torch

class DenseLayer(nn.Module):
    def __init__(self, inchannel, midchannel, outchannel):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm3d(inchannel)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.conv1_1 = nn.Conv3d(inchannel, midchannel, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm3d(midchannel)
        self.conv3_3 = nn.Conv3d(midchannel, outchannel, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.prelu(self.bn1(x))
        x = self.conv1_1(x)
        x = self.prelu(self.bn2(x))
        x = self.conv3_3(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DenseBlock, self).__init__()
        self.layer1 = DenseLayer(inchannel, 64, 16)
        self.layer2 = DenseLayer(16, 64, outchannel)
    def forward(self, x):
        x0 = x
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        res = torch.cat([x0, x1, x2], dim=1)
        return res

class TransitionLayer(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(TransitionLayer, self).__init__()
        self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv3d(outchannel, outchannel, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(inchannel)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.prelu(self.bn1(x))
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class TransitionLayer3(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(TransitionLayer3, self).__init__()
        self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm3d(inchannel)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.prelu(self.bn1(x))
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class NIDenseNet(nn.Module):
    def __init__(self):
        super(NIDenseNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)
        self.denseblock1 = DenseBlock(64, 16)
        self.translayer1 = TransitionLayer(64 + 16 + 16, 48)
        self.denseblock2 = DenseBlock(48, 16)
        self.translayer2 = TransitionLayer(48 + 16 + 16, 40)
        self.denseblock3 = DenseBlock(40, 16)
        self.translayer3 = TransitionLayer3(40 + 16 + 16, 36)
        self.denseblock4 = DenseBlock(36, 16)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(68, 5)

    #
    def forward(self, x):
        x = self.conv1(x)
        x = self.denseblock1(x)
        x = self.translayer1(x)
        x = self.denseblock2(x)
        x = self.translayer2(x)
        x = self.denseblock3(x)
        x = self.translayer3(x)
        x = self.denseblock4(x)
        x = self.avgpool(x)
        f = x
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        res = x
        # print(res.shape)
        # return res # stage1
        return f, res # stage2

if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_test = np.random.rand(1, 1, 128, 128, 128)
    inputs_test = torch.from_numpy(inputs_test)
    inputs_test = inputs_test.float()
    inputs_test = inputs_test.to(device)
    net_test = NIDenseNet().to(device)
    f, res = net_test(inputs_test)
    print(f.shape)
    # print(x)
    # print(x)