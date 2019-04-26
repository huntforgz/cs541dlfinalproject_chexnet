from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


    # def __init__(self, channels, reduction = 16):
    #     super(SELayer, self).__init__()
    #     self.avg_pool = nn.AdaptiveAvgPool2d(1)
    #     self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
    #                          padding=0)
    #     self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
    #                          padding=0)
    #     self.sigmoid = nn.Sigmoid()
    #
    # def forward(self, x):
    #     module_input = x
    #     x = self.avg_pool(x)
    #     x = self.fc1(x)
    #     x = nn.functional.relu(x)
    #     x = self.fc2(x)
    #     x = self.sigmoid(x)
    #     return module_input * x
    # 
