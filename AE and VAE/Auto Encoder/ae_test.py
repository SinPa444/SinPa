import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 32
                   # After runing train.py you have two save and model. path both of them
save_path = ''  
model_path = ''  

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat

model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
latent_vectors = torch.load(save_path).to(device)

with torch.no_grad():
    reconstructed_images = model(latent_vectors).view(-1, 28, 28).cpu()

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(reconstructed_images[i], cmap='gray')
    plt.title(f'Reconstructed {i+1}')
    plt.axis('off')
plt.show()