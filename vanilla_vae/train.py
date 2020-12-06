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

## def vae_loss(recon_x, x, mu, logvar, beta=5):
##     BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
##     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
##     return BCE + beta*KLD

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy_with_logits(recon, x)
    KL_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = recon_loss + 0.01 * KL_loss
    return loss

# vae params
learning_rate = 1e-3

epochs = 30
batch_size = 64

transform = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root="/scratch/anurag_deshmukh/data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="/scratch/anurag_deshmukh/data", train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

test_iter = iter(test_loader)

model = VAE()
classifier = nn.Sequential(nn.Flatten(), nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10), nn.Softmax(dim=-1))

#params = list(model.parameters()) + list(classifier.parameters())
model_optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
class_optim = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

now = datetime.now()
writer = SummaryWriter('./plots/vae_{}'.format(now.strftime("%d_%H_%M")))

def train(epoch):
    model.train()
    classifier.train()

    train_loss = 0
    v_loss = 0
    c_loss = 0

    train_acc = 0

    niter = 0
    
    criterion = nn.CrossEntropyLoss(reduction='sum')

    data_len = len(train_loader.dataset)
    loader_len = len(train_loader)

    # import pdb; pdb.set_trace();

    for index, (imgs, labels) in enumerate(train_loader):
        
        batch_len = len(imgs)

        model.zero_grad()
        recon_imgs, mu, logvar = model(imgs)
        v_loss = vae_loss(recon_imgs, imgs, mu, logvar)
        v_loss.backward()
        model_optim.step()

        class_optim.zero_grad()
        with torch.no_grad():
            comb = torch.cat([mu, logvar], dim=-1)
        probs = classifier(comb)
        c_loss = criterion(probs, labels)
        c_loss.backward()
        class_optim.step()

        ## for name, param in classifier.named_parameters():
        ##     if 'weight' in name:
        ##         param.data = torch.tanh(param.data)

        total_loss = v_loss.item() + c_loss.item()
        train_loss += total_loss
       
        preds = probs.argmax(dim=-1)
        train_acc += sum(preds == labels).item()
        
        writer.add_scalar('Train loss', total_loss/batch_len, niter + epoch*data_len)
        writer.add_scalar('VAE loss', v_loss/batch_len, niter + epoch*data_len)
        writer.add_scalar('Classification loss', c_loss/batch_len, niter + epoch*data_len)

        niter += batch_len

        writer.add_scalar('Train Accuracy', train_acc/niter, niter + epoch*data_len)
        

        if index%50 == 0:
            print("Epoch: {} [{:6}/{}]\t Avg. Loss: {:.3f} VAE Loss: {:.3f} Class Loss: {:.3f} Avg. Acc: {:.3f}".format(epoch, index*batch_len, data_len, total_loss/niter, v_loss/batch_len, c_loss, train_acc/niter))

def test(epoch):
    model.eval()
    classifier.eval()

    test_loss = 0
    v_loss = 0
    c_loss = 0

    test_acc = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for index, (imgs, labels) in enumerate(test_loader):
            recon_imgs, mu, logvar = model(imgs)
            comb = torch.cat([mu, logvar], dim=-1)
            probs = classifier(comb)

            v_loss += vae_loss(recon_imgs, imgs, mu, logvar)
            c_loss += criterion(probs, labels)
            total_loss = v_loss + c_loss
            test_loss += total_loss
     
            preds = probs.argmax(dim=-1)
            test_acc += sum(preds == labels).item()

            if index == 0:
                n = min(len(imgs), 16)
                comparison = torch.cat([imgs[:n], recon_imgs[:n]])
                writer.add_image('Recon Images', make_grid(comparison), epoch)
                # show(make_grid(comparison, padding=10))

    test_loss /= len(test_loader)
    v_loss /= len(test_loader)
    c_loss /= len(test_loader)
    test_acc /= len(test_loader.dataset)
    print("===============================>\t Avg. Test loss: {:.4f} VAE Loss: {:.4f} Class Loss: {:.4f} Test Acc: {:.4f}".format(test_loss, v_loss, c_loss, test_acc))

for epoch in range(epochs):
    train(epoch)
    test(epoch)
