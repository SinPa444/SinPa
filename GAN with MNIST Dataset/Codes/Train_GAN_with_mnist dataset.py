import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
    

def train_gan(save_path=''):    # Change this to your desired save path
    batch_size = 64
    n_epochs = 200
    lr_rate_G = 0.0002
    lr_rate_D = 0.0012
    latent_dim = 100


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    mnist = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)


    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_rate_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_rate_D, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()


    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)


            real_imgs = imgs.to(device)
            optimizer_D.zero_grad()
            real_validity = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_validity, real_label)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_validity, fake_label)


            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_validity = discriminator(fake_imgs)
            g_loss = adversarial_loss(fake_validity, real_label)
            g_loss.backward()
            optimizer_G.step()

            if i % 200 == 0:
                print(f"[epoch: {epoch}/{n_epochs}] [Batch: {i}/{len(dataloader)}]"
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                      f"[device: {device}]")
                

    torch.save(generator.state_dict(), save_path)
    print(f"Generator model saved to {save_path}")


if __name__ == "__main__":
    model_save_path = ''    # Change this to your desired save path
    print("Training GAN...")
    train_gan(save_path=model_save_path)
