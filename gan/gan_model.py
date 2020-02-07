from typing import List, Dict, Callable

import torch
from torch import Tensor, nn
from torch.nn import init

from gan.discriminator import Discriminator
from gan.loss.hinge import HingeLoss
from gan.loss.vanilla import DCGANLoss
from gan.loss.wasserstein import WassersteinLoss
from stylegan2_pytorch.model import Generator as StyleGenerator2
from stylegan2_pytorch.model import Discriminator as StyleDiscriminator2
from gan.generator import Generator
from gan.loss.gan_loss import GANLoss
from gan.loss_base import Loss
from optim.min_max import MinMaxParameters, MinMaxOptimizer, MinMaxLoss



def gan_weights_init(net, init_type='xavier', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class GANModel:

    def __init__(self, generator: Generator, loss: GANLoss, lr=0.0002, do_init_ws=True):
        self.generator = generator
        self.loss = loss
        if do_init_ws:
            self.generator.apply(gan_weights_init)
            self.loss.discriminator.apply(gan_weights_init)
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr, lr)

    def loss_pair(self, real: List[Tensor], *noise: Tensor) -> MinMaxLoss:

        fake = self.generator.forward(*noise)

        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return MinMaxLoss(
            self.loss.generator_loss(real, fake),
            self.loss.discriminator_loss_with_penalty(real, fake)
        )

    def generator_loss(self, real: List[Tensor], *noise: Tensor) -> Loss:
        fake = self.generator.forward(*noise)
        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return self.loss.generator_loss(real, fake)

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.loss.parameters())

    def forward(self, *noise: Tensor):
        return self.generator.forward(*noise)

    def train(self, real: List[Tensor], *noise: Tensor):
        loss = self.loss_pair(real, *noise)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()


class ConditionalGANModel(GANModel):

    def loss_pair(self, real: List[Tensor], condition: Tensor, *noise: Tensor) -> MinMaxLoss:

        fake = self.generator.forward(condition, *noise)

        if not isinstance(fake, (list, tuple)):
            fake = [fake]
        return MinMaxLoss(
            self.loss.generator_loss(real + [condition], fake + [condition]),
            self.loss.discriminator_loss_with_penalty(real + [condition], fake + [condition])
        )

    def generator_loss(self, real: List[Tensor], condition: Tensor, *noise: Tensor) -> Loss:
        return super().generator_loss(real, *([condition] + [*noise]))

    def forward(self, condition: Tensor, *noise: Tensor):
        return super().forward(*([condition] + [*noise]))

    def train(self, real: List[Tensor], condition: Tensor, *noise: Tensor):
        loss = self.loss_pair(real, condition, *noise)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()


name_to_gan_loss = {
    "hinge": lambda net_d: HingeLoss(net_d),
    "wasserstein": lambda net_d: WassersteinLoss(net_d, penalty_weight=10),
    "vanilla": lambda net_d: DCGANLoss(net_d)
}


class StyleGen2Wrapper(Generator):

    def __init__(self,
                 gen: StyleGenerator2,
                 return_latents=False,
                 inject_index=None,
                 truncation=1,
                 truncation_latent=None,
                 input_is_latent=False,
                 noise=None,
                 randomize_noise=False
                 ):
        super().__init__()
        self.gen: StyleGenerator2 = gen
        self.preproc = nn.Identity()

        self.return_latents = return_latents
        self.inject_index = inject_index
        self.truncation = truncation
        self.truncation_latent = truncation_latent
        self.input_is_latent = input_is_latent
        self.noise = noise
        self.randomize_noise = randomize_noise

    def forward(self, *input: Tensor):

        styles = self.preproc([*input])
        if not isinstance(styles, list):
            styles = [styles]

        img, _ = self.gen(styles,
                        self.return_latents,
                        self.inject_index,
                        self.truncation,
                        self.truncation_latent,
                        self.input_is_latent,
                        self.noise,
                        self.randomize_noise)

        return img


class StyleDisc2Wrapper(Discriminator):

    def __init__(self, disc: StyleDiscriminator2):
        super().__init__()
        self.disc = disc
        
    def forward(self, *img: Tensor):
        return self.disc([*img])


def stylegan2(path: str, loss_type: str, lr: float, noise=None) -> GANModel:

    state_dict = torch.load(path)

    generator: StyleGenerator2 = StyleGenerator2(256, 512, 8, channel_multiplier=2)
    generator.load_state_dict(state_dict['g'])
    generator = StyleGen2Wrapper(generator, noise=noise).cuda()

    discriminator: StyleDiscriminator2 = StyleDiscriminator2(256, 2)
    discriminator.load_state_dict(state_dict['d'])
    discriminator = StyleDisc2Wrapper(discriminator).cuda()

    loss: GANLoss = name_to_gan_loss[loss_type](discriminator)

    return GANModel(generator, loss, lr, do_init_ws=False)



