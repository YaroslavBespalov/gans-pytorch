from torch import nn, Tensor

from framework.gan.generator import Generator as G
from framework.nn.modules.common.self_attention import SelfAttention2d
from framework.nn.modules.resnet.residual import Down2xResidualBlock, PaddingType
from useful_utils.spectral_functions import spectral_norm_init


class DCEncoder(G):
    def __init__(self, nc: int = 3, nc_out: int = 10, ndf: int = 32):
        super(DCEncoder, self).__init__()

        self.main = nn.Sequential(
            spectral_norm_init(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # input is (ndf) x 64 x 64
            spectral_norm_init(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # state size. (ndf) x 32 x 32
            SelfAttention2d(ndf),
            spectral_norm_init(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 2, affine=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm_init(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 4, affine=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm_init(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = spectral_norm_init(nn.Linear(ndf * 8 * 4 * 4, nc_out))

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )
