"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3_gn_relu(in_c, out_c, kernel_size=3):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=1, stride=1, bias=False),
        nn.GroupNorm(out_c // 4, out_c),
        nn.LeakyReLU(inplace=True)
    )

def conv1_sigmoid(in_c, out_c, kernel_size=1):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, stride=1, bias=False),
        nn.Sigmoid()
    )

class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet_base(nn.Module):
    def __init__(self, n_channels, bilinear=True, chs=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(UNet_base, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])

    def forward(self, x):
        x1 = self.inc(x)
        print('after inc', x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class ConAttUNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=(16, 32, 64, 128, 256, 128, 64, 32, 16)):
        super(ConAttUNet, self).__init__()
        self.unet = UNet_base(n_channels=n_channels, chs=depth)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool3d((2, 2, 2)) # 2, 2, 2
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv_x5_3 = conv3_gn_relu(256, 16)
        self.conv_x5_1 = conv1_sigmoid(16, 1)

        self.conv_x4_3 = conv3_gn_relu(256, 16)
        self.conv_x4_1 = conv1_sigmoid(16, 1)

        self.conv_x3_3 = conv3_gn_relu(128, 16)
        self.conv_x3_1 = conv1_sigmoid(16, 1)

        self.conv_x2_3 = conv3_gn_relu(64, 16)
        self.conv_x2_1 = conv1_sigmoid(16, 1)

        self.conv_x1_3 = conv3_gn_relu(32, 16)
        self.conv_x1_1 = conv1_sigmoid(16, 1)

        self.sigmoid = nn.Sigmoid()

        chs = (16, 32, 64, 128, 256, 128, 64, 32, 16)
        self.up1 = Up(chs[4] + chs[3], chs[5], True)
        self.up2 = Up(chs[5] + chs[2], chs[6], True)
        self.up3 = Up(chs[6] + chs[1], chs[7], True)
        self.up4 = Up(chs[7] + chs[0], chs[8], True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)


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
        # Encoder
        en_x1, en_x2, en_x3, en_x4, en_x5 = self.unet(x)

        # Decoder
        de_x4 = self.up1(en_x5, en_x4)
        de_x3 = self.up2(de_x4, en_x3)
        de_x2 = self.up3(de_x3, en_x2)
        de_x1 = self.up4(de_x2, en_x1)
        print('encoder', en_x1.shape, en_x2.shape, en_x3.shape, en_x4.shape, en_x5.shape)
        print('decoder', de_x1.shape, de_x2.shape, de_x3.shape, de_x4.shape)

        # self-check1
        x5_mul = self.conv_x5_1(self.conv_x5_3(en_x5))
        # print(x5_mul.shape)
        x5_mul = self.upsample(x5_mul)
        # print(x5_mul.shape)

        # self-check2
        # print(torch.cat([de_x4, en_x4]).shape, self.conv_x4_3(torch.cat([de_x4, en_x4])).shape,
        #       self.conv_x4_1(self.conv_x4_3(torch.cat([de_x4, en_x4]))).shape)
        print(torch.cat([de_x4, en_x4], dim=1).shape)
        x4_mul = self.conv_x4_1(self.conv_x4_3(torch.cat([de_x4, en_x4], dim=1)))
        print(x4_mul.shape)
        x4_mul = x4_mul * x5_mul
        x4_mul = self.upsample(x4_mul)
        # print(x4_mul.shape)

        # self-check3
        x3_mul = self.conv_x3_1(self.conv_x3_3(torch.cat([de_x3, en_x3], dim=1)))
        x3_mul = x3_mul * x4_mul
        x3_mul = self.upsample(x3_mul)
        # print(x3_mul.shape)

        # self-check4
        x2_mul = self.conv_x2_1(self.conv_x2_3(torch.cat([de_x2, en_x2], dim=1)))
        x2_mul = x2_mul * x3_mul
        x2_mul = self.upsample(x2_mul)
        # print(x2_mul.shape)

        # self-check5
        x1_mul = self.conv_x1_1(self.conv_x1_3(torch.cat([de_x1, en_x1], dim=1)))
        # print('----\n', x1_mul.shape)
        # print(en_x1.shape)
        # print(x2_mul.shape)
        x1_mul = x1_mul * x2_mul
        # print(x1_mul.shape)

        return x1_mul[:, :, diffZ // 2: Z + diffZ // 2, diffY // 2: Y + diffY // 2,
                                  diffX // 2:X + diffX // 2]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_test = np.random.rand(1, 1, 128, 128, 128)
    inputs_test = torch.from_numpy(inputs_test)
    inputs_test = inputs_test.float()
    inputs_test = inputs_test.to(device)
    net_test = ConAttUNet(n_channels=1, n_classes=1).to(device)
    x = net_test(inputs_test)
    print(x.shape, x.max())
