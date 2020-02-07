import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.gan.loss.gan_loss import GANLoss


class DCGANLoss(GANLoss):

    __criterion = nn.BCELoss()

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        batch_size = dgz.size(0)
        nc = dgz.size(1)

        real_labels = torch.full((batch_size, nc, ), 1, device=dgz.device)
        errG = self.__criterion(dgz.view(batch_size, nc).sigmoid(), real_labels)
        return Loss(errG)

    def discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:

        batch_size = dx.size(0)
        nc = dx.size(1)

        real_labels = torch.full((batch_size, nc, ), 1, device=dx.device)
        err_real = self.__criterion(dx.view(batch_size, nc).sigmoid(), real_labels)

        fake_labels = torch.full((batch_size, nc, ), 0, device=dx.device)
        err_fake = self.__criterion(dy.view(batch_size, nc).sigmoid(), fake_labels)

        return Loss(-(err_fake + err_real))

