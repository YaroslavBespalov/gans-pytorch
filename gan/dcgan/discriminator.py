import torch
from torch import nn, Tensor

from dataset.probmeasure import ProbabilityMeasureFabric
from framework.gan.conditional import ConditionalDiscriminator
from framework.gan.discriminator import Discriminator as D
from framework.nn.modules.common.self_attention import SelfAttention2d
from framework.nn.modules.resnet.residual import Down2xResidualBlock, PaddingType
from useful_utils.spectral_functions import spectral_norm_init


class DCDiscriminator(D):
    def __init__(self, nc: int = 3, nc_out: int = 1, ndf: int = 32):
        super(DCDiscriminator, self).__init__()

        self.main = nn.Sequential(
            spectral_norm_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),n_power_iterations=10),
            nn.InstanceNorm2d(ndf, affine=True), #nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),n_power_iterations=10),
            nn.InstanceNorm2d(ndf, affine=True), #nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # input is (ndf) x 64 x 64
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),n_power_iterations=10),
            nn.InstanceNorm2d(ndf, affine=True),#nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # state size. (ndf) x 32 x 32
            SelfAttention2d(ndf),
            spectral_norm_init(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 2, affine=True), #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 2, affine=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm_init(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 4, affine=True), #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 4, affine=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm_init(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 8, affine=True), #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_init(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 8, affine=True),  # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = spectral_norm_init(nn.Linear(ndf * 8 * 2 * 2, nc_out), n_power_iterations=10)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )


class ConditionalDCDiscriminator(ConditionalDiscriminator):

    def __init__(self, nc: int = 4, nc_out: int = 1, ndf: int = 32):
        super(ConditionalDCDiscriminator, self).__init__()

        self.main = nn.Sequential(
            spectral_norm_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf), #nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # input is (ndf) x 64 x 64
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # state size. (ndf) x 32 x 32
            SelfAttention2d(ndf),
            spectral_norm_init(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 2, affine=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm_init(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 4, affine=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm_init(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = spectral_norm_init(nn.Linear(ndf * 8 * 4 * 4, nc_out), n_power_iterations=10)

    def forward(self, x: Tensor, *condition: Tensor) -> Tensor:
        cond = ProbabilityMeasureFabric(x.shape[-1]).from_channels(condition[0][:, 0:600]).toImage(x.shape[-1])
        x = torch.cat([x, cond], dim=1)
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )



class ResDCDiscriminator(D):
    def __init__(self, nc=3, ndf=32):
        super(ResDCDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # Down2xResidualBlock(nc, ndf, PaddingType.REFLECT, nn.InstanceNorm2d,
            #                     nn.LeakyReLU(0.2, inplace=True), use_spectral_norm=True),
            spectral_norm_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # Down2xResidualBlock(ndf, ndf, PaddingType.REFLECT, nn.InstanceNorm2d,
            #                     nn.LeakyReLU(0.2, inplace=True), use_spectral_norm=True),
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False), n_power_iterations=10),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention2d(ndf),
            Down2xResidualBlock(ndf, ndf * 2, PaddingType.REFLECT, nn.InstanceNorm2d,
                                nn.LeakyReLU(0.2, inplace=True), use_spectral_norm=True),
            Down2xResidualBlock(ndf * 2, ndf * 4, PaddingType.REFLECT, nn.InstanceNorm2d,
                                nn.LeakyReLU(0.2, inplace=True), use_spectral_norm=True),
            Down2xResidualBlock(ndf * 4, ndf * 8, PaddingType.REFLECT, nn.InstanceNorm2d,
                                nn.LeakyReLU(0.2, inplace=True), use_spectral_norm=True),
        )

        self.linear = spectral_norm_init(nn.Linear(ndf * 8 * 4 * 4, 1, bias=False), n_power_iterations=10)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )
