import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, mid_channels):
        super(AttentionBlock, self).__init__()
        self.W_skip = nn.Sequential(nn.Conv3d(skip_channels, mid_channels, kernel_size=1),
                                    nn.BatchNorm3d(mid_channels))
        self.W_x = nn.Sequential(nn.Conv3d(in_channels, mid_channels, kernel_size=1),
                                 nn.BatchNorm3d(mid_channels))
        self.psi = nn.Sequential(nn.Conv3d(mid_channels, 1, kernel_size=1),
                                 nn.BatchNorm3d(1),
                                 nn.Sigmoid())

    def forward(self, x_skip, x):
        x_skip = self.W_skip(x_skip)
        x = self.W_x(x)
        out = self.psi(nn.ReLU(inplace=True)(x_skip + x))
        return out * x_skip


class AttentionUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AttentionUp, self).__init__()
        self.attention = AttentionBlock(in_ch, out_ch, out_ch)
        self.conv1 = conv_block(in_ch+out_ch, out_ch)

    def forward(self, x, x_skip):
        # note : x_skip is the skip connection and x is the input from the previous block
        x = nn.functional.interpolate(x, x_skip.shape[2:], mode='trilinear', align_corners=False)
        x_attention = self.attention(x_skip, x)
        # stack their channels to feed to both convolution blocks
        x = torch.cat((x, x_attention), dim=1)
        x = self.conv1(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AttentionUNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up_conv5 = AttentionUp(filters[4], filters[3])

        self.Up_conv4 = AttentionUp(filters[3], filters[2])

        self.Up_conv3 = AttentionUp(filters[2], filters[1])

        self.Up_conv2 = AttentionUp(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up_conv5(e5,e4)

        d3 = self.Up_conv4(d4,e3)

        d2 = self.Up_conv3(d3,e2)

        d1 = self.Up_conv2(d2,e1)

        out = self.Conv(d1)

        return out


if __name__ == '__main__':
    x = torch.randn(2,1,96,96,48)
    model = AttentionUNet(in_ch=1,out_ch=2)
    y = model(x)
    print(y.shape)