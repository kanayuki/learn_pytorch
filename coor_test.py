import torch
from torch import nn
from torchsummary import summary


class CoorClassifier(nn.Module):
    def __init__(self):
        super(CoorClassifier, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(True),

            nn.Linear(5, 10),
            nn.ReLU(True),

            nn.Linear(10, 4),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


def train(model):
    optimizer = torch.optim.SGD(model.parameters(), 1e-2)
    loss_fn = nn.CrossEntropyLoss().cuda()

    max_iter = 1000

    for i in range(max_iter):
        x = 1
        y = 1
        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"第{i}次，loss: {loss.item()}")


if __name__ == '__main__':
    model = CoorClassifier().cuda()
    summary(model, (1, 2))
