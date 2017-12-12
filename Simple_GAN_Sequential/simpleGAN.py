import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, help='mnist')
parser.add_argument('--dataroot', default='../../MNIST_data', help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='out-pytorch-gan', help='folder to output images and model checkpoints')

opt = parser.parse_args()

print("Options: ", opt)

try:
	os.makedirs(opt.outf)
except:
	pass

# set the min batch size
mb_size = opt.batchSize 

# the number of samples to take from a random distribution
Z_dim = opt.nz 

# The number of training examples
X_dim = opt.imageSize * opt.imageSize

# The size of hidden layer
h_dim = 128

# Set the learning rate
lr = opt.lr

#load data
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(opt.dataroot, train=True, download=True,
    	transform=transforms.Compose([
    		transforms.Scale(opt.imageSize),
    		transforms.ToTensor()
    	])),
    batch_size=mb_size, shuffle=True
)  

# Lets define the 'Xavier Initialization function'
def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        in_dim = m.weight.data.size()[1]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        m.weight.data.normal_(0.0, xavier_stddev)
        m.bias.data.fill_(0)


def plot_loss(g_loss, d_loss):
    plt.plot(g_loss)
    plt.savefig('{}/gen_loss.png'.format(opt.outf), bbox_inches='tight')

    plt.plot(d_loss)
    plt.savefig('{}/dis_loss.png'.format(opt.outf), bbox_inches='tight')

#Generator Network
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(Z_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output = self.main(x)
        
        return output

#Discriminator Network
class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(X_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output = self.main(x)
        
        return output

#Get instance of both classes
G = Generator()
D = Discriminator()

# Apply custom weights
G.apply(xavier_init)
D.apply(xavier_init)

if opt.netG != '':
	G.load_state_dict(torch.load(opt.netG))

if opt.netD != '':
	D.load_state_dict(torch.load(opt.netD))

# Get parameters
G_params = G.parameters()
D_params = D.parameters()

# Reset gradients for Generator and Discriminator
def reset_grad(netG, netD):
    netG.zero_grad()
    netD.zero_grad()

# Define loss
criterion = nn.BCEWithLogitsLoss()
# Declare the optimizers for generator
G_solver = optim.Adam(G_params, lr=1e-3)
# Declare the optimizers for discriminator
D_solver = optim.Adam(D_params, lr=1e-3)

# The targets for the discriminator.
# Zeros for the samples coming from generator and one for the data coming from sample space
ones_label = Variable(torch.ones(mb_size))
zeros_label = Variable(torch.zeros(mb_size))

# resize to fit loss criterion
ones_label.data.resize_([mb_size, 1])
zeros_label.data.resize_([mb_size, 1])

reset_grad(G, D)
gen_loss = []
dis_loss = []

for epoch in range(opt.nepoch):
    for it, (data, _) in enumerate(dataloader):
        
        # Sample data
        z = Variable(torch.randn(mb_size, Z_dim))
        X = Variable(data.resize_([mb_size, X_dim]))

        # Dicriminator forward-loss-backward-update
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss_real = criterion(D_real, ones_label)
        D_loss_fake = criterion(D_fake, zeros_label)
        D_loss = D_loss_real + D_loss_fake

        D_loss.backward()
        D_solver.step()

        # Housekeeping - reset gradient
        reset_grad(G, D)

        # Generator forward-loss-backward-update
        z = Variable(torch.randn(mb_size, Z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = criterion(D_fake, ones_label)

        G_loss.backward()
        G_solver.step()

        # Housekeeping - reset gradient
        reset_grad(G, D)
        
        # Print and plot every now and then
        if it % 1000 == 0:
            print('Epoch-{}; D_loss: {}; G_loss: {}'.format(epoch, D_loss.data.numpy(), G_loss.data.numpy()))
            dis_loss.append(D_loss.data.numpy())
            gen_loss.append(G_loss.data.numpy())

            samples = G(z).data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            plt.savefig('{}/epoch-{}.png'.format(opt.outf, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

            plot_loss(gen_loss, dis_loss)

    #save model
    torch.save(G.state_dict(), '%s/G_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(D.state_dict(), '%s/D_epoch_%d.pth' % (opt.outf, epoch))
