import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock1(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.bn = nn.BatchNorm3d(num_features=outchannel)
        self.dropout = nn.Dropout()
    def forward(self, x):
        x = self.conv1(x)
        x1 = x
        x = self.prelu(x)
        x = self.bn(x)
        x = self.dropout(self.conv1(x))
        x = x + x1
        return x

class ResBlock2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResBlock2, self).__init__()
        self.resblock1 = ResBlock1(inchannel, outchannel)
        self.resblock2 = ResBlock1(inchannel, outchannel)

    def forward(self, x):
        x1 = self.resblock1(x)
        x1 = x + x1
        x2 = self.resblock2(x1)
        res = x + x2
        return res


class NINet(nn.Module):
    def __init__(self):
        super(NINet, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.resblock1_1 = ResBlock1(inchannel=16, outchannel=16)
        self.downsample1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.resblock1_2 = ResBlock1(inchannel=32, outchannel=32)
        self.downsample2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.resblock2_1 = ResBlock2(inchannel=64, outchannel=64)
        self.downsample3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.resblock2_2 = ResBlock2(inchannel=128, outchannel=128)

        self.upsample1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.resblock2_3 = ResBlock2(inchannel=64, outchannel=64)
        self.upsample2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.resblock1_3 = ResBlock1(inchannel=32, outchannel=32)
        self.upsample3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.resblock1_4 = ResBlock1(inchannel=16, outchannel=16)

        self.conv2 = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=1)
        self.simoid = nn.Sigmoid()
        self.adaptivemaxpool = nn.AdaptiveMaxPool3d((2, 2, 2))
        self.classifier2 = nn.Sequential(
            nn.Linear(1920, 5),
            # nn.LeakyReLU(inplace=True)#五分类时改成了5，其余没变
        )

    def forward(self, x):
        Z = x.size()[2]
        Y = x.size()[3]
        X = x.size()[4]
        diffZ = (16 - x.size()[2] % 16) % 16
        diffY = (16 - x.size()[3] % 16) % 16
        diffX = (16 - x.size()[4] % 16) % 16
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        x = self.conv1(x)
        x = self.resblock1_1(x)
        x1 = x
        x = self.downsample1(x)
        x = self.resblock1_2(x)
        x2 = x
        x = self.downsample2(x)
        x = self.resblock2_1(x)
        x3 = x
        x = self.downsample3(x)
        x = self.resblock2_2(x)
        f1 = self.adaptivemaxpool(x)
        f1 = f1.view(f1.shape[0], -1)

        x = self.upsample1(x)
        x = x + x3
        x = self.resblock2_3(x)
        f2 = self.adaptivemaxpool(x)
        f2 = f2.view(f2.shape[0], -1)
        x = self.upsample2(x)
        x = x + x2
        x = self.resblock1_3(x)
        f3 = self.adaptivemaxpool(x)
        f3 = f3.view(f3.shape[0], -1)
        x = self.upsample3(x)
        x = x + x1
        x = self.resblock1_4(x)
        f4 = self.adaptivemaxpool(x)
        f4 = f4.view(f4.shape[0], -1)
        x = self.conv2(x)
        x = self.simoid(x)
        seg_res = x

        f = torch.cat([f1, f2, f3, f4], dim=1)
        cls = self.classifier2(f)

        return f, seg_res[:, :, diffZ // 2: Z + diffZ // 2, diffY // 2: Y + diffY // 2,
                        diffX // 2:X + diffX // 2]# stage2
        # return seg_res[:, :, diffZ // 2: Z + diffZ // 2, diffY // 2: Y + diffY // 2,
        #        diffX // 2:X + diffX // 2], cls # stage1


if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_test = np.random.rand(1, 1, 128, 128, 128)
    inputs_test = torch.from_numpy(inputs_test)
    inputs_test = inputs_test.float()
    inputs_test = inputs_test.to(device)
    net_test = NINet().to(device)
    x, cls = net_test(inputs_test)
    print(x.shape)
    print(x.max(), x.min(), cls)