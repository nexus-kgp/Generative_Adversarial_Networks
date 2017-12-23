import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
from utils import merge_images
from models import Gxy, Gyx
from models import Dx, Dy


class CycleGAN(object):
	def __init__(self, config, svhn_loader, mnist_loader):
		self.svhn_loader = svhn_loader
		self.mnist_loader = mnist_loader

		self.Gxy = None
		self.Gyx = None
		self.Dx = None
		self.Dy = None

		self.G_optim = None
		self.D_optim = None

		self.G_conv_dim = config.g_conv_dim
		self.D_conv_dim = config.d_conv_dim

		#self.use_reconst_loss = config.use_reconst_loss true
		self.beta1 = config.beta1
		self.beta2 = config.beta2
		self.train_iters = config.train_iters
		self.batch_size = config.batch_size
		self.lr = config.lr

		self.cuda = config.cuda
		self.log_step = config.log_step
		self.sample_step = config.sample_step
		self.sample_path = config.sample_path
		self.model_path = config.model_path

		self.build_model()

	def build_model(self):
		
		self.Gxy = Gxy(conv_dim=self.G_conv_dim)
		self.Gyx = Gyx(conv_dim=self.G_conv_dim)

		self.Dx = Dx(conv_dim=self.D_conv_dim)
		self.Dy = Dy(conv_dim=self.D_conv_dim)

		G_params = list(self.Gxy.parameters()) + list(self.Gyx.parameters())
		D_params = list(self.Dx.parameters()) + list(self.Dy.parameters())

		self.G_optim = optim.Adam(G_params, self.lr, [self.beta1, self.beta2])
		self.D_optim = optim.Adam(D_params, self.lr, [self.beta1, self.beta2])

		if torch.cuda.is_available() and self.cuda:
			#TODO: CUDA
			pass

	def np_to_var(self, x):
		if torch.cuda.is_available() and self.cuda :
			x = x.cuda()
		return Variable(x)

	def var_to_np(self, x):
		if torch.cuda.is_available() and self.cuda:
			x = x.cpu()
		return x.data.numpy()

	def reset_grad(self):
		self.G_optim.zero_grad()
		self.D_optim.zero_grad()

	def train(self):
		svhn_iter = iter(self.svhn_loader)
		mnist_iter = iter(self.mnist_loader)
		iter_per_epoch = min(len(svhn_iter), len(mnist_iter))

		fixed_svhn =  self.np_to_var(svhn_iter.next()[0])
		fixed_mnist =  self.np_to_var(mnist_iter.next()[0])

		for step in range(self.train_iters+1):
			if (step+1) % iter_per_epoch == 0:
				mnist_iter = iter(self.mnist_loader)
				svhn_iter = iter(self.svhn_loader)

			svhn, _ = svhn_iter.next()
			svhn = self.np_to_var(svhn)
			mnist, _ = mnist_iter.next()
			mnist = self.np_to_var(mnist)

			#============ train D ============#
			# real images
			self.reset_grad()

			out = self.Dx(mnist)
			Dx_loss = torch.mean((out-1)**2)

			out = self.Dy(svhn)
			Dy_loss = torch.mean((out-1)**2)

			D_real_loss = Dx_loss + Dy_loss
			D_real_loss.backward()

			self.D_optim.step()

			# fake images
			self.reset_grad()

			out = self.Dy(self.Gxy(mnist))
			Dy_loss = torch.mean(out**2)

			out = self.Dx(self.Gyx(svhn))
			Dx_loss = torch.mean(out**2)

			D_fake_loss = Dx_loss + Dy_loss
			D_fake_loss.backward()

			self.D_optim.step()

			#============ train G ============#
			# mnist-svhn-mnist cycle
			self.reset_grad()

			mnist_to_svhn = self.Gxy(mnist)
			out = self.Dy(mnist_to_svhn)
			mnist_reconst = self.Gyx(mnist_to_svhn)

			# adversarial loss
			G_loss = torch.mean((out-1)**2)
			# cycle-consistency loss
			G_loss += torch.mean((mnist - mnist_reconst)**2)

			G_loss.backward()
			self.G_optim.step()

			# svhn-mnist-svhn cycle
			self.reset_grad()

			svhn_to_mnist = self.Gyx(svhn)
			out = self.Dx(svhn_to_mnist)
			svhn_reconst = self.Gxy(svhn_to_mnist)

			# adversarial loss
			G_loss = torch.mean((out-1)**2)
			# cycle-consistency loss
			G_loss += torch.mean((svhn - svhn_reconst)**2)

			G_loss.backward()
			self.G_optim.step()

			# print logs
			if (step+1) % self.log_step == 0:
				print('Step [%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f'
					% (step+1, self.train_iters, D_real_loss.data[0], D_fake_loss.data[0], G_loss.data[0]))

			if (step+1) % self.sample_step == 0:
				fake_mnist = self.Gyx(fixed_svhn)
				fake_svhn = self.Gxy(fixed_mnist)

				mnist, fake_mnist = self.var_to_np(fixed_mnist), self.var_to_np(fake_mnist)
				svhn , fake_svhn = self.var_to_np(fixed_svhn), self.var_to_np(fake_svhn)

				merged = merge_images(mnist, fake_svhn)
				path = os.path.join(self.sample_path, 'sample-%d-m-s.png' % (step+1))
				scipy.misc.imsave(path, merged)

				print('Saved %s' % path)

				merged = merge_images(svhn, fake_mnist)
				path = os.path.join(self.sample_path, 'sample-%d-s-m.png' % (step+1))
				scipy.misc.imsave(path, merged)

				print('Saved %s' % path)

			if (step+1) % 5000 == 0:
				Gxy_path = os.path.join(self.model_path, 'Gxy-%d.pkl' % (step+1))
				Gyx_path = os.path.join(self.model_path, 'Gyx-%d.pkl' % (step+1))

				Dx_path = os.path.join(self.model_path, 'Dx-%d.pkl' % (step+1))
				Dy_path = os.path.join(self.model_path, 'Dy-%d.pkl' % (step+1))

				torch.save(self.Gxy.state_dict(), Gxy_path)
				torch.save(self.Gyx.state_dict(), Gyx_path)
				torch.save(self.Dx.state_dict(), Dx_path)
				torch.save(self.Dy.state_dict(), Dy_path)

	def sample(self):
		#TODO
		pass
