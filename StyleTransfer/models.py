import torch.nn as nn
import torch.nn.functional as F
from utils import conv_layer, deconv_layer

class Gxy(nn.Module):
	def __init__(self, conv_dim=64):
		super(Gxy, self).__init__()

		# encoding block
		self.conv1 = conv_layer(1, conv_dim, 4)
		self.conv2 = conv_layer(conv_dim, conv_dim*2, 4)

		# residual block (will use res layer when I have good GPU :P)
		self.conv3 = conv_layer(conv_dim*2, conv_dim*2, 3, 1, 1)
		self.conv4 = conv_layer(conv_dim*2, conv_dim*2, 3, 1, 1)

		# docoding block
		self.deconv1 = deconv_layer(conv_dim*2, conv_dim, 4)
		self.deconv2 = deconv_layer(conv_dim, 3, 4, batch_norm=False)

	def forward(self, x):
		out = F.leaky_relu(self.conv1(x), 0.05)
		out = F.leaky_relu(self.conv2(out), 0.05)

		out = F.leaky_relu(self.conv3(out), 0.05)
		out = F.leaky_relu(self.conv4(out), 0.05)

		out = F.leaky_relu(self.deconv1(out), 0.05)
		out = F.tanh(self.deconv2(out))

		return out

class Gyx(nn.Module):
	def __init__(self, conv_dim=64):
		super(Gyx, self).__init__()

		# encoding block
		self.conv1 = conv_layer(3, conv_dim, 4)
		self.conv2 = conv_layer(conv_dim, conv_dim*2, 4)

		# residual block
		self.conv3 = conv_layer(conv_dim*2, conv_dim*2, 3, 1, 1)
		self.conv4 = conv_layer(conv_dim*2, conv_dim*2, 3, 1, 1)

		# docoding block
		self.deconv1 = deconv_layer(conv_dim*2, conv_dim, 4)
		self.deconv2 = deconv_layer(conv_dim, 1, 4, batch_norm=False)

	def forward(self, x):
		out = F.leaky_relu(self.conv1(x), 0.05)
		out = F.leaky_relu(self.conv2(out), 0.05)

		out = F.leaky_relu(self.conv3(out), 0.05)
		out = F.leaky_relu(self.conv4(out), 0.05)

		out = F.leaky_relu(self.deconv1(out), 0.05)
		out = F.tanh(self.deconv2(out))

		return out

class Dx(nn.Module):
	def __init__(self, conv_dim=64):
		super(Dx, self).__init__()

		self.conv1 = conv_layer(1, conv_dim, 4, batch_norm=False)
		self.conv2 = conv_layer(conv_dim, conv_dim*2, 4)
		self.conv3 = conv_layer(conv_dim*2, conv_dim*4, 4)

		self.fc = conv_layer(conv_dim*4, 1, 4, 1, 0, batch_norm=False)

	def forward(self, x):
		out = F.leaky_relu(self.conv1(x), 0.05)
		out = F.leaky_relu(self.conv2(out), 0.05)
		out = F.leaky_relu(self.conv3(out), 0.05)
		out = self.fc(out).squeeze()

		return out

class Dy(nn.Module):
	def __init__(self, conv_dim=64):
		super(Dy, self).__init__()

		self.conv1 = conv_layer(3, conv_dim, 4, batch_norm=False)
		self.conv2 = conv_layer(conv_dim, conv_dim*2, 4)
		self.conv3 = conv_layer(conv_dim*2, conv_dim*4, 4)

		self.fc = conv_layer(conv_dim*4, 1, 4, 1, 0, batch_norm=False)

	def forward(self, x):
		out = F.leaky_relu(self.conv1(x), 0.05)
		out = F.leaky_relu(self.conv2(out), 0.05)
		out = F.leaky_relu(self.conv3(out), 0.05)
		out = self.fc(out).squeeze()

		return out