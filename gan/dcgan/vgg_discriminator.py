from torch import nn, Tensor

from framework.gan.discriminator import Discriminator as D
from framework.nn.modules.common.View import View
from framework.nn.modules.common.vgg import Vgg16, Vgg19BN
from framework.parallel import ParallelConfig
from useful_utils.spectral_functions import spectral_norm_init


class VGGDiscriminator(D):
    def __init__(self):
        super(VGGDiscriminator, self).__init__()
        nc = 3
        ndf = 64

        depth = 17

        self.vgg = Vgg16(depth)

        self.main = nn.Sequential(
            # state size. (512) x 16 x 16
            spectral_norm_init(nn.Conv2d(256, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm_init(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            View(-1, ndf * 8 * 4 * 4),
            spectral_norm_init(nn.Linear(ndf * 8 * 4 * 4, 10))
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

        # self.linear = spectral_norm_init(nn.Linear(ndf * 8 * 4 * 4, 10))

    def forward(self, x: Tensor) -> Tensor:
        x = self.vgg(x)
        # print(vgg.shape)
        conv = self.main(x)
        return conv
