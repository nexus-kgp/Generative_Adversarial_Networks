# Generative_Adversarial_Networks
Generative adversarial networks (GANs) are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework. They were introduced by Ian Goodfellow in 2014.

The branch `plain_pytorch_implementation` contains the PyTorch implementation of the same.

* [See the live action in your browser](https://cs.stanford.edu/people/karpathy/gan/)
* [Intersting Medium Read](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f)


Results : 
* **Before start of training** : 

![Before start of training](./out-pytorch-gan/000.png)
* **After one iteration** : 

![After one iteration](./out-pytorch-gan/001.png)
* **After two iterations** : 

![After two iteration](./out-pytorch-gan/002.png)
* **After some hundred thousands iterations** : 

![After some hundred thousands iterations](./out-pytorch-gan/102.png)


**Generator's Loss** : ![gen-loss](././out-pytorch-gan/disloss.png)

**Discriminator's Loss** : ![disc-loss](././out-pytorch-gan/genloss.png)


### Well here are some results from the Deep Convolutional GAN on CIFAR-10 dataset

![GIF of epochs](./DCGAN_Results/dcgan_cifar10.gif)