import torch
import torch.nn as nn
import numpy

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class Fcvt(nn.Module):
    def __init__(self,input_dim=3,out_dim=1):
        super(Fcvt, self).__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=1),
            nn.ReLU(inplace=True)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
            )
        self.att = eca_layer(64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(67, 67, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(67, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.att(x)
        x = torch.cat([residual,x],dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x