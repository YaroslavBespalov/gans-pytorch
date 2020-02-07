import torch
from torch import nn, Tensor
import numpy as np

from framework.nn.modules.resnet.down_block import DownBlock
from framework.nn.modules.resnet.residual import ResidualNet, PaddingType
from framework.nn.modules.resnet.up_block import UpBlock
from framework.gan.conditional import ConditionalGenerator
from framework.gan.noise import Noise


class ResidualGenerator(ConditionalGenerator):

    def __init__(self,
                 noise: Noise,
                 input_nc: int,
                 output_nc: int,
                 ngf: int = 64,
                 n_down: int = 3,
                 n_residual_blocks: int = 9,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 padding_type: PaddingType = PaddingType.REFLECT):
        super(ResidualGenerator, self).__init__(noise)

        self.n_down = n_down
        max_ngf = 1024
        nz = noise.size()

        self.model_downsample = DownBlock(input_nc, min(ngf * (2 ** n_down), max_ngf), ngf, n_down, norm_layer)

        self.model_resnet = ResidualNet(min(ngf * (2 ** n_down), max_ngf) + nz, n_residual_blocks, norm_layer, padding_type)

        self.model_upsample = UpBlock(min(ngf * (2 ** n_down), max_ngf) + 2 * nz, output_nc, ngf, n_down, norm_layer)

        self.noise_upsample = nn.Sequential(
            nn.ConvTranspose2d(nz, nz, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz),
            nn.ReLU(inplace=True)
        )

    def noise_to_matrix(self, noise: Tensor, size: int) -> Tensor:
        assert int(2 ** np.log2(size)) == size
        tmp_size = 1
        tmp_noise = noise.view(noise.shape[0], self.noise_gen.size(), 1, 1)
        while tmp_size < size:
            tmp_noise = self.noise_upsample(tmp_noise)
            tmp_size *= 2
        return tmp_noise

    def _forward_impl(self, noise: Tensor, condition: Tensor) -> Tensor:

        assert condition.shape[-1] == condition.shape[-2]
        noise_matrix = self.noise_to_matrix(noise, int(condition.shape[-1] / (2 ** self.n_down)))

        downsample = self.model_downsample(condition)
        resnet = self.model_resnet(torch.cat([downsample, noise_matrix], dim=1))
        upsample = self.model_upsample(torch.cat([resnet, noise_matrix], dim=1))

        return upsample


