import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

## class VAE(nn.Module):
##     
##     def __init__(self):
##         super(VAE, self).__init__()
## 
##         self.fc1 = nn.Linear(28*28, 400)
##         self.fc21 = nn.Linear(400, 128)
##         self.fc22 = nn.Linear(400, 128)
##         self.fc3 = nn.Linear(128, 512)
##         self.fc4 = nn.Linear(512, 28*28)
## 
##         #self.classifier = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10), nn.Softmax(dim=-1))
## 
##     def reparameterize(self, mu, logvar):
##         std = torch.exp(0.5*logvar)
##         eps = torch.randn_like(std)
##         return mu + eps*std
## 
##     def encode(self, x):
##         out = F.relu(self.fc1(x))
##         return self.fc21(out), self.fc22(out)
## 
##     def decode(self, z):
##         out = F.relu(self.fc3(z))
##         return torch.sigmoid(self.fc4(out))
## 
##     def forward(self, x):
##         mu, logvar = self.encode(x.view(-1, 28*28))
##         z = self.reparameterize(mu, logvar)
##         #comb = torch.cat([mu, logvar], dim=-1)
##         return self.decode(z), mu, logvar

class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_down, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale,scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x

    
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_up, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    
class Encoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)#64
        self.conv2 = Res_down(ch, 2*ch)#32
        self.conv3 = Res_down(2*ch, 4*ch)#16
        self.conv4 = Res_down(4*ch, 8*ch)#8
        self.conv5 = Res_down(8*ch, 8*ch)#4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)#2
        self.conv_logvar = nn.Conv2d(8*ch, z, 2, 2)#2

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x, Train = True):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if Train:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            x = self.sample(mu, logvar)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        return x, mu, logvar
    
class Decoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, ch*8)
        self.conv2 = Res_up(ch*8, ch*8)
        self.conv3 = Res_up(ch*8, ch*4)
        self.conv4 = Res_up(ch*4, ch*2)
        self.conv5 = Res_up(ch*2, ch)
        self.conv6 = Res_up(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 
    
class VAE(nn.Module):
    def __init__(self, channel_in = 3, z = 512):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(channel_in, z = z)
        self.decoder = Decoder(channel_in, z = z)

    def forward(self, x, Train = True):
        encoding, mu, logvar = self.encoder(x, Train)
        recon = self.decoder(encoding)
        return recon, mu, logvar
