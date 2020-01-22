from torch import nn, optim
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

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

    batch = 12

    args = parser.parse_args()

    device = 'cuda'

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.path)['g_running'])
    generator.eval()

    generator_train = StyledGenerator(512).to(device)
    generator_train.load_state_dict(torch.load(args.path)['g_running'])
    generator_train.train()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2

    # z, img = sample(generator, step, mean_style, batch, device)

    # print(z.shape)

    noise = []

    for i in range(step + 1):
        size = 4 * 2 ** i
        noise.append(torch.zeros(batch, 1, size, size, device=device))

    inv_gen = InvGenarator().to(device)

    inv_gen = nn.DataParallel(inv_gen)

    opt = optim.Adam(inv_gen.parameters(), lr=0.001)
    opt_gen = optim.Adam(generator_train.parameters(), lr=0.0001)

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = ImageFolder("../../face", transform)
    loader = data.DataLoader(dataset, batch_size=batch)

    mariya, _ = iter(loader).__next__()
    mariya = mariya.to(device)

    for i in range(10000):

        z, img = sample(generator, step, mean_style, batch, device, noise)
        img = img.detach()
        z_pred = inv_gen(img)

        img_pred = generator_train(
            [z_pred, z_pred],
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
            noise=noise
        )

        loss = nn.L1Loss()(img_pred, img.detach())

        generator_train.zero_grad()
        inv_gen.zero_grad()
        loss.backward()
        opt.step()
        opt_gen.step()

        if i % 10 == 0:
            z_m = inv_gen(mariya)

            mariya_pred = generator_train(
                [z_m, z_m],
                step=step,
                alpha=1,
                mean_style=mean_style,
                style_weight=0.7,
                noise=noise
            )

            loss = nn.L1Loss()(mariya_pred, mariya.detach())

            generator_train.zero_grad()
            inv_gen.zero_grad()
            loss.backward()
            opt.step()
            opt_gen.step()

        if i % 100 == 0:
            print(i, loss.item())

            with torch.no_grad():
                z_m = inv_gen(mariya)

                mariya_pred = generator_train(
                    [z_m, z_m],
                    step=step,
                    alpha=1,
                    mean_style=mean_style,
                    style_weight=0.7,
                    noise=noise
                )

                utils.save_image(
                    torch.cat([mariya, mariya_pred], 0), f'sample_inv.png', nrow=batch, normalize=True, range=(-1, 1)
                )

