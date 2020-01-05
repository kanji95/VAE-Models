import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from models import VAE

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} device is being used!!".format(device))

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# vae params
learning_rate = 1e-3

epochs = 100
batch_size = 128

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

test_iter = iter(test_loader)

model = VAE()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.KLDivLoss()

for epoch in range(epochs):

    model.train()

    for ind, (images, labels) in enumerate(train_loader):
        output, mu, log_sigma = model(images)
        
        optimizer.zero_grad()

        loss = loss_function(output, images, mu, log_sigma)
        loss.backward()

        optimizer.step()
        
        if ind%10 == 0:
            print("Epoch {} [{}/{}] Total loss {}".format(epoch, ind, len(train_loader), loss/len(images)))

    if epoch%10 == 0:

        image, label = next(test_iter)
        image = image.reshape(-1, 784)

        model.eval()

        output, _, _ = model(image)
        output = output.reshape(-1, 1,   28, 28)

        output = output[:32]
        show(make_grid(output, padding=10))
