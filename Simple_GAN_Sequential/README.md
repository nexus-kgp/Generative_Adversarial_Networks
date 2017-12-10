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

**Discriminator's Loss** : ![disc-loss](./out-pytorch-gan/dis_loss.png)
