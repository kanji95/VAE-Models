import torch
import torch.nn as nn

class VAE(nn.Module):
    
    def __init__(self, in_features, latent_size, batch_size):
        super(VAE, self).__init__()
        self.enc_fc_1 = nn.Linear(in_features, 512)
        self.enc_fc_2 = nn.Linear(512, 256)

        self.latent_size = latent_size

        self.mu = nn.Linear(256, latent_size)
        self.log_sigma = nn.Linear(256, latent_size)
        
        self.epsilon = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        # self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=2)
        # self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
    
    def forward(self, input_):
        input_ = input_.reshape(-1, 784)
        print("{} 0th level output shape".format(input_.shape))
        out = self.enc_fc_1(input_)
        print("{} 1st level output shape".format(out.shape))
        out = self.enc_fc_2(out)
        print("{} 2nd level output shape".format(out.shape))

        mu = self.mu(out)
        log_sigma = self.log_sigma(out)

        eps = self.epsilon.sample(self.latent_size)
        print(eps.shape)
        return mu, log_sigma
        # print("{} 0th level output shape".format(input_.shape))
        # out = self.conv_1(input_)
        # print("{} 1st level output shape".format(out.shape))
        # out = self.conv_2(out)
        # print("{} 2nd level output shape".format(out.shape))
        # return torch.flatten(out)
