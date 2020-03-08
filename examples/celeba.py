from __future__ import print_function
#%matplotlib inline
import random
import time

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torchvision import utils

from gan.conjugate_gan_model import ConjugateGANModel
from gan.dcgan.discriminator import DCDiscriminator, PosDCDiscriminator
from gan.dcgan.generator import DCGenerator
from gan.image2image.residual_generator import ResidualGenerator
from gan.loss.penalties.conjugate import ConjugateGANLoss
from gan.noise.normal import NormalNoise
from models.grad import Grad

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batch_size = 16
image_size = 128
noise_size = 100

dataset = dset.ImageFolder(root="/raid/data/celeba",
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=12)

device = torch.device("cuda")
noise = NormalNoise(noise_size, device)
netG = DCGenerator(noise_size, image_size).to(device)
netD = PosDCDiscriminator(image_size).to(device)
netT = Grad(PosDCDiscriminator(image_size)).to(device)

gan_model = ConjugateGANModel(netG, ConjugateGANLoss(netD, netT), lr=0.001, do_init_ws=False)

print("Starting Training Loop...")

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)
        z = noise.sample(batch_size)

        loss_d = gan_model.train_disc([imgs], z)

        loss_g = 0
        if i % 5 == 0 and i > 0:
            loss_g = gan_model.train_gen(z)

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss_d, loss_g))

        if i % 100 == 0:
            # with torch.no_grad():
                fake = gan_model.forward(z).detach().cpu()
                utils.save_image(
                    fake, f'sample_{i}.png', nrow=batch_size//2, normalize=True, range=(-1, 1)
                )
