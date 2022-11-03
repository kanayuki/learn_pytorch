import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

batch_size = 128
num_epochs=5
learning_rate=1e-3

train_data = MNIST(r"\..\data", train=True, transform=transforms.ToTensor(), download=True)
test_data = MNIST(r"\..\data", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(True),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


net = Net().cuda()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), learning_rate)

# Train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).cuda()
        labels = labels.cuda()
        outputs = net(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]  Step [{i + 1}/{total_step}]  Loss: {loss.item()}')

# Test
with torch.no_grad():
    total = 0
    correct = 0
    for (images, labels) in test_loader:
        images = images.reshape(-1, 28 * 28).cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("在{}张测试图像上网络的准确度为：{}%".format(total, correct / total * 100))

# Save
torch.save(net.state_dict(), 'feedforward.ckpt')
