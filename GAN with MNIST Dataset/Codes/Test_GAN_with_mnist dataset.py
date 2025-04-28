import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn


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
    
def display_images(images, n_cols=4, figsize=(8, 2), title='Images'):
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=12)
    for i, img in enumerate(images):
        plt.subplot(1, n_cols, i+1)
        plt.imshow(img[0].cpu().numpy(), cmap="gray")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_dataset_images(num_images=4):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=num_images, shuffle=True)


    images, _ = next(iter(train_loader))
    print('Displaying MNIST dataset images...')
    display_images(images, n_cols=num_images, figsize=(8, 2), title='MNIST Dataset Images')


def test_random_images(model_path='', num_images=4):       # Change the path to your model file
    latent_dim = 100
    generator = Generator().to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        return
    except RuntimeError as e:
        print(f"Error: Could not load model. Ensure the model architecture matvhes the saved model. Detaols: {e}")
        return
    
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        fake_images = generator(z)
    print('Displaying generator images...')
    display_images(fake_images, n_cols=num_images, figsize=(8, 2), title='Generated Images')


if __name__ == '__main__':
    model_path = ""   # Change the path to your model file

    show_dataset_images(num_images=4)

    test_random_images(model_path=model_path, num_images=4)
