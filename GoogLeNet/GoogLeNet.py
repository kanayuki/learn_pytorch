import torchvision
import torch
from torch import nn
from torchsummary import summary


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, ceil_mode=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


if __name__ == '__main__':
    # my_model = GoogLeNet().cuda()
    # summary(my_model, (3, 255), 255)

    model = torchvision.models.GoogLeNet().cuda()
    summary(model, (3, 255, 255))
