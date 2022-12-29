import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), padding='same'),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), padding='same'),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='valid'),
            nn.ELU(),
            nn.Dropout(0.2)
        )
        self.lin = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=1),
            nn.ELU()
        )

    def forward(self, input):
        out = self.classifier(input)
        out = out.view((128, ))
        out = self.lin(out)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(in_features=37, out_features=64),
            nn.ELU(),
            nn.Linear(in_features=64, out_features=480),
            nn.ELU()
        )
        self.gen = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same'),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 2), padding='valid'),
            nn.ELU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, padding='valid'),
            nn.ELU()
        )

    def forward(self, input):
        out = self.lin(input)
        out = out.view((3, 4, 40))
        out = self.gen(out)
        out = out.view((10, 15))
        return out
