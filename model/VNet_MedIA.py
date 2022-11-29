import torch.nn as nn
import torch.nn.functional as F
import torch


class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # with learnable parameters
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.conv_3 = conv3d(out_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        # print('conv_x3', z_3.shape, z_1.shape, (z_3 + self.skip_connection(x)).shape)
        return z_3 + self.skip_connection(x), z_3


class conv3d_x2(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x2, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        return z_2 + self.skip_connection(x)


class conv3d_x1(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x1, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        # print('Conv3d_x1', z_1.shape, self.skip_connection(x).shape)
        return z_1 + self.skip_connection(x)


class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels , out_channels)
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        # print('deconv_x3', self.conv_x3(rhs_add).shape, rhs_up.shape)
        return self.conv_x3(rhs_add) + rhs_up, self.conv_x3(rhs_add)


class deconv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x2, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels, out_channels)
        self.conv_x2 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x2(rhs_add) + rhs_up


class deconv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x1, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels, out_channels)
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x1(rhs_add) + rhs_up


def conv3d_as_pool(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
        nn.ReLU())


def deconv3d_as_up(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        nn.ReLU()
    )


class sigmoid_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(sigmoid_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        return nn.Sigmoid()(y_conv)
        # return y_conv


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.conv_1 = conv3d_x1(1, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x2(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)

        self.bottom = conv3d_x3(256, 256)

        self.deconv_4 = deconv3d_x3(256, 128)
        self.deconv_3 = deconv3d_x3(128, 64)
        self.deconv_2 = deconv3d_x2(64, 32)
        self.deconv_1 = deconv3d_x1(32, 16)

        self.out = sigmoid_out(16, 1)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.cls = nn.Sequential(
            nn.Linear(128+256+128, 64),
            nn.Linear(64, 5),
            # nn.Sigmoid()
        )
        self.gn128 = nn.GroupNorm(128, 128)
        self.gn256 = nn.GroupNorm(256, 256)

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
        # print(x.shape)

        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3, _ = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        conv_4, fea1 = self.conv_4(pool)
        pool = self.pool_4(conv_4)
        # print('fea1', fea1.shape)
        fea1 = self.gap(fea1)
        # fea1 = self.gap(self.gn128(fea1))
        bottom, fea2 = self.bottom(pool)
        # print('fea2', fea2.shape)
        fea2 = self.gap(fea2)
        # fea2 = self.gap(self.gn256(fea2))
        deconv, fea3 = self.deconv_4(conv_4, bottom)
        # print('fea3', fea3.shape)
        fea3 = self.gap(fea3)
        # fea3 = self.gap(self.gn128(fea3))
        deconv, _ = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        seg_res = self.out(deconv)

        fea = torch.cat([fea1, fea2, fea3], dim=1)
        fea = fea.view(fea.size(0), -1)
        cls_res = self.cls(fea)

        return seg_res[:, :, diffZ // 2: Z + diffZ // 2, diffY // 2: Y + diffY // 2,
                                  diffX // 2:X + diffX // 2], cls_res

if __name__ == '__main__':
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs_test = np.random.rand(2, 1, 128, 128, 128)
    inputs_test = torch.from_numpy(inputs_test)
    inputs_test = inputs_test.float()
    inputs_test = inputs_test.to(device)
    net_test = VNet().to(device)
    x, y = net_test(inputs_test)
    print(x)
    print(x.shape, x.min(), x.max(), y, y.shape)

