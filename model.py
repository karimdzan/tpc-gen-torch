import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same', bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 3, 1, 0)
        )
        self.apply(weights_init)

    def forward(self, input):
        out = self.classifier(input)
        out = out.view((input.shape[0], -1))
        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, bias=False),
            nn.Tanh()
        )
        self.output_linear = nn.Linear(in_features=169, out_features=150)
        self.apply(weights_init)

    def forward(self, input):
        input = input.unsqueeze(-1).unsqueeze(-1)
        out = self.gen(input)
        out = out.view((-1, 169))
        out = self.output_linear(out)
        out = out.view((-1, 1, 10, 15))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)