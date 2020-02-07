from torch import nn, Tensor

from framework.gan.discriminator import Discriminator as D
from useful_utils.spectral_functions import spectral_norm_init

class EDiscriminator(D):
    def __init__(self, dim=2, ndf=64):
        super(EDiscriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(dim, ndf), # spectral_norm_init(nn.Linear(dim, ndf)),
            nn.ReLU(True),
            nn.Linear(ndf, 2 * ndf),# spectral_norm_init(nn.Linear(ndf, 2 * ndf)),
            nn.ReLU(True),
            nn.Linear(2 * ndf, 2 * ndf), # spectral_norm_init(nn.Linear(2 * ndf, 2 * ndf)),
            nn.ReLU(True),
            nn.Linear(2 * ndf, 1) # spectral_norm_init(nn.Linear(2 * ndf, 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
