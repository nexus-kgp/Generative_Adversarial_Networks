import argparse
import os
from cycleGAN import CycleGAN
from utils import get_loader


def main(config):
	shvn_loader, mnist_loader = get_loader(config)

	cgan = CycleGAN(config, shvn_loader, mnist_loader)

	if not os.path.exists(config.model_path):
		os.makedirs(config.model_path)
	if not os.path.exists(config.sample_path):
		os.makedirs(config.sample_path)

	if config.mode == 'train':
		cgan.train()
	elif config.mode == 'sample':
		cgan.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # misc
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--mnist_path', type=str, default='./mnist')
    parser.add_argument('--svhn_path', type=str, default='./svhn')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)

    config = parser.parse_args()
    print(config)
    main(config)