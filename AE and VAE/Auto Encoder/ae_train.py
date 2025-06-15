import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
latent_dim = 32
epochs = 100
learning_rate = 0.001
                     # path the folder that you want to save your model and vectors
save_path = ''  #  
model_path = '' #

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

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
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        output, _ = model(data)
        loss = criterion(output, data.view(-1, 28 * 28))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)

model.eval()
latent_vectors = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        _, z = model(data)
        latent_vectors.append(z.cpu())
latent_vectors = torch.cat(latent_vectors, dim=0)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(latent_vectors, save_path)

print(f"vectors saved in {save_path}")
print(f"model saved in {model_path}")