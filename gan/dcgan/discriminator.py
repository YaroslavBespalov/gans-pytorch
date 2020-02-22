import torch
from torch import nn, Tensor
from gan.discriminator import Discriminator as D
from models.attention import SelfAttention2d
from models.positive import PosConv2d, PosLinear


class DCDiscriminator(D):
    def __init__(self, image_size: int, nc: int = 3, nc_out: int = 1, ndf: int = 32):
        super(DCDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        tmp_size = image_size // 2
        tmp_nc = ndf
        while tmp_size > 2:
            tmp_size = tmp_size // 2
            nc_next = min(256, tmp_nc * 2)
            layers += [
                nn.Conv2d(tmp_nc, nc_next, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(nc_next, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if tmp_size == 32:
                layers += [SelfAttention2d(nc_next)]
            tmp_nc = nc_next

        self.main = nn.Sequential(*layers)

        self.linear = nn.Linear(ndf * 8 * 2 * 2, nc_out)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )


class PosDCDiscriminator(D):
    def __init__(self, image_size: int, nc: int = 3, nc_out: int = 1, ndf: int = 64):
        super(PosDCDiscriminator, self).__init__()

        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(ndf, affine=True),
        ]

        tmp_size = image_size // 2
        tmp_nc = ndf
        nc_next = -1
        while tmp_size > 2:
            tmp_size = tmp_size // 2
            nc_next = min(256, tmp_nc * 2)
            layers += [
                PosConv2d(tmp_nc, nc_next, 4, 2, 1),
                nn.CELU(0.2, inplace=True),
                nn.InstanceNorm2d(nc_next, affine=True),
            ]
            # if tmp_size == 32:
            #     layers += [SelfAttention2d(nc_next)]
            tmp_nc = nc_next

        self.main = nn.Sequential(*layers)

        self.linear = PosLinear(nc_next * 2 * 2, nc_out)

    def forward(self, x: Tensor) -> Tensor:
        conv = self.main(x)
        return self.linear(
            conv.view(conv.shape[0], -1)
        )



