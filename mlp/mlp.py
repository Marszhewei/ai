import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False


if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     
    transform=torchvision.transforms.ToTensor(),    
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(28*28,28*28),
            nn.Linear(28*28,10)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output, x 

mlp = MLP()


optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28*28)

        output = mlp(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = mlp(test_x.view(-1,28*28))
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


test_output, _ = mlp(test_x[:100].view(-1,28*28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
pred_succ = 0
pred_fail = 0
for (pred_val, test_val) in zip(pred_y, test_y[:100].numpy()):
    if pred_val == test_val:
        pred_succ += 1
        print("predict succ, the num is {}, is odd: {}".format(pred_val, pred_val % 2))
    else:
        pred_fail += 1
        print("predict failed.")
print("predict succ rate: {}%".format(pred_succ/(pred_succ+pred_fail)*100))

