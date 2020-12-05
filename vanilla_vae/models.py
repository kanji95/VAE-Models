import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, 128)
        self.fc22 = nn.Linear(400, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 28*28)

        #self.classifier = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10), nn.Softmax(dim=-1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        out = F.relu(self.fc1(x))
        return self.fc21(out), self.fc22(out)

    def decode(self, z):
        out = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(out))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        #comb = torch.cat([mu, logvar], dim=-1)
        return self.decode(z), mu, logvar
