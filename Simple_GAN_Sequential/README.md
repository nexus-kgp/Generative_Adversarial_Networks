# Generative_Adversarial_Networks using nn.Sequential model

Generator Model:

`Generator (
  (main): Sequential (
    (0): Linear (100 -> 128)
    (1): ReLU (inplace)
    (2): Linear (128 -> 784)
    (3): Sigmoid ()
  )
)`

Discriminator Model:

`Discriminator (
  (main): Sequential (
    (0): Linear (784 -> 128)
    (1): ReLU (inplace)
    (2): Linear (128 -> 1)
    (3): Sigmoid ()
  )
)`

## Usage (to run python file)
```
usage: simpleGAN.py [-h] [--dataroot DATAROOT] [--batchSize BATCHSIZE]
                    [--imageSize IMAGESIZE] [--nz NZ] [--nepoch NEPOCH]
                    [--lr LR] [--netG NETG] [--netD NETD] [--outf OUTF]

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   path to dataset
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --nepoch NEPOCH       number of epochs to train for
  --lr LR               learning rate, default=0.001
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
```

Results : 
* **Before start of training** : 

![Before start of training](./out-pytorch-gan/000.png)
* **After one iteration** : 

![After one iteration](./out-pytorch-gan/001.png)
* **After two iterations** : 

![After two iteration](./out-pytorch-gan/002.png)
* **After a hundred thousand iterations** : 

![After a hundred thousand iterations](./out-pytorch-gan/099.png)


**Generator's Loss** : ![gen-loss](./out-pytorch-gan/gen_loss.png)

**Discriminator's Loss** : ![dis-loss](./out-pytorch-gan/dis_loss.png)
