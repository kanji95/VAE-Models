import torch
import torchvision
import torchvision.transforms as transforms
from models import VAE

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("{} device is being used!!".format(device))

# vae params
latent_size = 100
learning_rate = 1e-3

epochs = 100
batch_size = 2
in_channels = 784

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# for ind, (image, label) in enumerate(train_loader):
#     print(image.shape, label.shape, label)

#     img = image[0].permute(1, 2, 0)
#     print(img.shape)

#     np_img = img.numpy().squeeze()
#     plt.imshow(np_img)
#     plt.show()
#     break
### 

train_iter = iter(train_loader)
image, label = next(train_iter)

model = VAE(in_channels, latent_size)

output = model(image)
print(output)



# print(model.parameters)
# optimizer = torch.optim.Adam(lr=learning_rate, model.parameters())