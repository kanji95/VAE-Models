import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import VAE

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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

    return BCE + 5*KLD

# vae params
learning_rate = 1e-3

epochs = 10
batch_size = 128

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

test_iter = iter(test_loader)

model = VAE()

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

now = datetime.now()
writer = SummaryWriter('./plots/vae_{}'.format(now.strftime("%d_%H_%M")))

def train(epoch):
    model.train()
    train_loss = 0
    niter = 0
    for index, (imgs, _) in enumerate(train_loader):
        optim.zero_grad()
        recon_imgs, mu, logvar = model(imgs)
        loss = loss_function(recon_imgs, imgs, mu, logvar)
        train_loss += loss.item()
        loss.backward()
        optim.step()
        
        writer.add_scalar('Train loss', loss.item()/len(imgs), niter + epoch*len(train_loader.dataset))
        niter += len(imgs)

        if index%50 == 0:
            print("Epoch: {} [{:6}/{} ({:3.2f}%)]\t Avg. Loss: {:.3f}".format(epoch, index*len(imgs), len(train_loader.dataset), 100. * index/len(train_loader), loss.item()/len(imgs)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for index, (imgs, _) in enumerate(test_loader):
            recon_imgs, mu, logvar = model(imgs)
            test_loss += loss_function(recon_imgs, imgs, mu, logvar)

            if index == 0:
                n = min(len(imgs), 16)
                comparison = torch.cat([imgs[:n], recon_imgs.view(-1, 1, 28, 28)[:n]])
                writer.add_image('Recon Images', make_grid(comparison), epoch)
                # show(make_grid(comparison, padding=10))

    test_loss /= len(test_loader.dataset)
    print("===============================>\t Avg. Test loss: {:.4f}".format(test_loss))

for epoch in range(epochs):
    train(epoch)
    test(epoch)
