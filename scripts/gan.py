import torch
from torch import nn
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train(data, generator, discriminator, num_epochs=200, batch_size=32):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for n_batch, real_batch in enumerate(data_loader):
            real_data = Variable(real_batch)
            current_batch_size = real_data.size(0)

            # Train discriminator
            d_optimizer.zero_grad()
            prediction_real = discriminator(real_data)
            error_real = criterion(prediction_real, torch.full((current_batch_size, 1), 1., dtype=torch.float32))

            noise = Variable(torch.randn(current_batch_size, generator.input_dim))
            fake_data = generator(noise)
            prediction_fake = discriminator(fake_data.detach())
            error_fake = criterion(prediction_fake, torch.full((current_batch_size, 1), 0., dtype=torch.float32))

            d_error = error_real + error_fake
            d_error.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            noise = Variable(torch.randn(current_batch_size, generator.input_dim))
            fake_data = generator(noise)
            prediction = discriminator(fake_data)
            g_error = criterion(prediction, torch.full((current_batch_size, 1), 1., dtype=torch.float32))

            g_error.backward()
            g_optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Discriminator Loss: {d_error.item()}")
        print(f"Generator Loss: {g_error.item()}")