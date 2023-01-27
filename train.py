import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from model import Discriminator, Generator
from tqdm import tqdm
from data.preprocessing import read_csv_2d
from metrics import plotting
from utils import LoadData
from scalers import get_scaler
import torchvision.utils as vutils
import os


class Trainer(object):

    def __init__(self, epochs=2000, num_disc_updates=5, batch_size=32, latent_dim=32, scaler='identity'):
        self.step_counter = 0
        self.NUM_DISC_UPDATES = num_disc_updates
        self.device = torch.device('cuda:0')
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.epochs = epochs
        self.transform = T.Compose([
            T.ToTensor()
        ])
        self.scaler = get_scaler(scaler_type=scaler)
        self.data, self.features = read_csv_2d(filename='data/data_v2/csv/digits.csv', strict=False)
        self.data = self.scaler.scale(self.data)
        self.data = np.float32(self.data)
        self.data = LoadData(self.data, self.transform)
        self.dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=4, pin_memory=True)
        # self.features = torch.Tensor(self.features)
        self.features = np.float32(self.features)
        # self.features = self.transform(self.features)
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    # def preprocess_features(self, features):
    #     print(features.shape)
    #     bin_fractions = torch.tensor(features[:, 2:4] % 1)
    #     features = (features[:, :3] - np.array([[0.0, 0.0, 162.5]]).reshape(-1, 1)) / np.array([[20.0, 60.0, 127.5]]).reshape(-1, 1)
    #     return torch.cat((features, bin_fractions), -1)

    def gen_step(self, noise):

        self.generator.zero_grad()

        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images)
        errG = -torch.mean(fake_output)
        D_G_z2 = fake_output.mean().item()
        errG.backward()
        self.optimizer_g.step()
        return errG

    def disc_step(self, batch, noise):
        self.discriminator.zero_grad()

        real_output = self.discriminator(batch)
        errD_real = torch.mean(real_output)
        D_x = real_output.mean().item()

        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images.detach())
        errD_fake = torch.mean(fake_output)
        D_G_z1 = fake_output.mean().item()
        gradient_penalty = self.calculate_penalty(self.discriminator,
                                                  batch.data, fake_images.data,
                                                  self.device)

        errD = -errD_real + errD_fake + gradient_penalty * 10
        errD.backward()

        self.optimizer_d.step()
        return errD

    def calculate_penalty(self, model, real, fake, device):
        alpha = torch.randn((real.size(0), 1, 1, 1), device=device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)

        model_interpolates = model(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    def train(self):
        self.discriminator.train()
        self.generator.train()
        loss_history = {'disc_losses': [], 'gen_losses': []}
        for epoch in range(self.epochs):
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for i, data in progress_bar:
                # print(data.shape)
                # data = data.view((self.batch_size, 1, 10, 15))
                real_images = data.to(self.device)
                batch_size = real_images.size(0)
                noise = torch.normal(0, 1, size=(batch_size, self.latent_dim), device=self.device)
                # noise = torch.cat((
                #     torch.tensor(self.features[batch_size * i:i * batch_size + batch_size, :], device=self.device), noise), -1)

                disc_loss = self.disc_step(real_images, noise)

                if (i + 1) % self.NUM_DISC_UPDATES == 0:
                    gen_loss = self.gen_step(noise)

                    loss_history['disc_losses'].append(disc_loss)
                    loss_history['gen_losses'].append(gen_loss)

            # print("epoch:   ", epoch)
            # print("disc_loss:    ", torch.mean(torch.tensor(loss_history['disc_losses'])))
            # print("gen_loss:    ", torch.mean(torch.tensor(loss_history['gen_losses'])))

            if (epoch + 1) % 500 == 0:
                vutils.save_image(real_images,
                                  os.path.join("output", "real_samples.png"),
                                  normalize=True)
                fake = self.generator(noise)
                vutils.save_image(fake.detach(),
                                  os.path.join("output", f"fake_samples_{epoch + 1}.png"),
                                  normalize=True)

            if (epoch + 1) % 4 == 0:
                plotting.plot_metrics(loss_history['gen_losses'], loss_history['disc_losses'])
        plotting.plot_metrics(loss_history['gen_losses'], loss_history['disc_losses'])


