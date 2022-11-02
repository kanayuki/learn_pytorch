import torch
from torch import nn
from torchsummary import summary
import torchvision


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),

            nn.AdaptiveAvgPool2d(6),

            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),

            nn.Linear(4096, 1000)

        )

    def forward(self, x):
        x = self.module(x)
        return x


if __name__ == '__main__':
    # my_model = AlexNet()
    # summary(my_model, (3, 224,224))

    alex_net = torchvision.models.AlexNet().cuda()
    summary(alex_net, (3, 224, 224))
