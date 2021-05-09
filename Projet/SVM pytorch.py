import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)


#rint(mnist_trainset[:][0])
#x=mnist_trainset[:] [0]
#y=mnist_trainset[:] [1]

x_train=mnist_trainset.data
y_train=mnist_trainset.classes


# Affichage de certaines images
fig, sub = plt.subplots(10, 15)

print(sub.flatten())
for i, ax in zip(range(0,300,2),sub.flatten()):
    ax.imshow(x_train[i+1000],cmap="gray")
    ax.axis('off')
plt.show()



import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)