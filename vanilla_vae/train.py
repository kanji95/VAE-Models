import torch
import torchvision
import torchvision.transforms as transforms
from models import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} device is being used!!".format(device))

# vae params
latent_size = 1000
learning_rate = 1e-3

epochs = 100
batch_size = 64

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

### 

model = VAE()