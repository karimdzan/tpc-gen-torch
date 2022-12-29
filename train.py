import torch
import numpy as np
from model import Discriminator, Generator
from tqdm import tqdm
from data.preprocessing import read_csv_2d
from metrics.plotting import plot_metrics


class Trainer(object):
    def __init__(self, epochs=1000, num_disc_updates=5, batch_size=64, latent_dim=32):
        self.step_counter = 0
        self.NUM_DISC_UPDATES = num_disc_updates
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.generator = self.generator.apply(self.init_weights)
        self.discriminator = self.discriminator.apply(self.init_weights)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.01, betas=(0.5, 0.999))
        self.epochs = epochs
        self.dataloader_train = LoadData()
        self.data, self.features = read_csv_2d(filename='data/data_v4/csv/digits.csv', strict=False)
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def preprocess_features(self, features):
        bin_fractions = features[:, 2:4] % 1
        features = (features[:, :3] - torch.tensor([[0.0, 0.0, 162.5]])) / torch.tensor([[20.0, 60.0, 127.5]])
        return torch.cat((features, bin_fractions), -1)

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

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.xavier_normal_(m.weight, 1.0)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.xavier_normal_(m.weight, 1.0)
            torch.nn.init.zeros_(m.bias)

    def train(self):
        self.discriminator.train()
        self.generator.train()
        loss_history = {'disc_losses': [], 'gen_losses': []}
        shuffle_ids = np.random.permutation(len(self.data))
        for epoch in range(self.epochs):
            for i, data in tqdm(range(0, len(self.data), self.batch_size)):
                batch = self.data[shuffle_ids][i : i + self.batch_size]
                real_images = batch.to(self.device)
                batch_size = real_images.size(0)
                noise = torch.normal(0, 1, size=(batch_size, self.latent_dim), device=self.device)
                noise = torch.cat((self.preprocess_features(self.features), noise), -1)

                disc_loss = self.disc_step(real_images, noise)

                if (i + 1) % self.NUM_DISC_UPDATES == 0:
                    gen_loss = self.gen_step(noise)

                    loss_history['disc_losses'].append(disc_loss)
                    loss_history['gen_losses'].append(gen_loss)

            print("epoch:   ", epoch)
            print("disc_loss:    ", np.mean(loss_history['disc_losses']))
            print("gen_loss:    ", np.mean(loss_history['gen_losses']))

        plot_metrics(loss_history['gen_losses'], loss_history['disc_loss'])


