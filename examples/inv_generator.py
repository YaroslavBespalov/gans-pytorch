from torch import nn, optim
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from gan.gan_model import stylegan2
from optim.min_max import MinMaxParameters, MinMaxOptimizer
from style_based_gan_pytorch.model import StyledGenerator, ConvBlock, EqualLinear
import argparse
import math
import torch
from torchvision import utils

from stylegan2_pytorch.dataset import MultiResolutionDataset


@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device, noise):
    z = torch.randn(n_sample, 512).to(device)
    image = generator(
        z,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
        noise=noise
    )

    return z, image


class InvGenarator(nn.Module):

    def __init__(self):
        super().__init__()

        code_dim = 512

        self.progression = nn.Sequential(
                ConvBlock(3, 128, 3, 1, downsample=True, fused=True),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=True),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                # ConvBlock(512, 512, 3, 1, 4, 0)
        )

        self.linear = nn.Sequential(
            # nn.Dropout(),
            EqualLinear(code_dim * 4 * 4, code_dim),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(code_dim, code_dim),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(code_dim, code_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(code_dim, code_dim),
            # nn.LeakyReLU(0.2, inplace=True),
            # EqualLinear(code_dim, code_dim),
            # nn.LeakyReLU(0.2, inplace=True),
            # EqualLinear(code_dim, code_dim),
        )

    def forward(self, img):
        return self.linear(self.progression(img).view(img.shape[0], -1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256, help='size of the image')
    parser.add_argument('path', type=str, help='path to checkpoint file')

    batch = 4

    args = parser.parse_args()
    size = 256

    device = 'cuda'

    noise = []
    step = int(math.log(args.size, 2)) - 2

    gan_model = stylegan2("/home/ibespalov/stylegan2/stylegan2-pytorch/checkpoint/130000.pt", "wasserstein", 0.0001)
    params = MinMaxParameters(gan_model.generator.gen.parameters(), gan_model.loss.discriminator.disc.final_linear.parameters())
    gan_model.optimizer = MinMaxOptimizer(params, 0.0001, 0.0004)

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=5, scale=(1, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = ImageFolder("../../face", transform)
    loader = data.DataLoader(dataset, batch_size=batch, shuffle=True)

    for epoch in range(1000):
        print(epoch)

        for i, (img, label) in enumerate(loader):
            img = img.cuda()

            z = torch.randn(img.shape[0], 512).to(device)
            loss_g, loss_d = gan_model.train([img], z)

        if epoch % 10 == 0:

                with torch.no_grad():

                    z = torch.randn(batch, 512).to(device)
                    mariya_pred = gan_model.forward(z)

                    utils.save_image(
                        mariya_pred, f'sample_inv.png', nrow=batch, normalize=True, range=(-1, 1)
                    )

