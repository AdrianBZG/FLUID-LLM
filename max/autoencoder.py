import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from dataloader.MGN_dataloader import MGNDataloader
from dataloader.parallel_dataloader import ParallelDataGenerator


class Autoencoder64(nn.Module):
    def __init__(self):
        super(Autoencoder64, self).__init__()
        self.act = nn.ELU()

        # Encoder
        self.enc_1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)  # Input: (3, 32, 32) Output: (16, 16, 16)
        self.enc_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # Output: (32, 8, 8)
        self.enc_3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # Output: (64, 4, 4)
        # self.enc_4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # Output: (128, 2, 2)
        # Decoder
        # self.dec_1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        self.dec_3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec_4 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.act(self.enc_1(x))
        x = self.act(self.enc_2(x))
        x = self.act(self.enc_3(x))
        # x = self.act(self.enc_4(x))

        # x = self.act(self.dec_1(x))
        x = self.act(self.dec_2(x))
        x = self.act(self.dec_3(x))
        x = self.dec_4(x)

        return x


def train_autoencoder():
    EPOCHS = 1000
    BATCH_SIZE = 64

    model = Autoencoder64().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

    ds = MGNDataloader(load_dir="./ds/MGN/cylinder_dataset",
                       resolution=512, patch_size=(32, 32), stride=(32, 32))
    dataloader = ParallelDataGenerator(ds, num_producers=4, queue_maxsize=8)
    dataloader.run()

    st = time.time()
    for epoch in range(EPOCHS):

        # Load data
        data, mask = dataloader.get()
        idx = torch.randint(0, data.shape[-1], [BATCH_SIZE])
        data = data[:, :, :, idx].cuda()
        data = torch.permute(data, [3, 0, 1, 2])
        mask = mask[:, :, idx].cuda()
        mask = torch.permute(mask, [2, 0, 1])
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, 3, 1, 1)

        # Forward pass
        output = model(data)
        error = (data - output)[torch.logical_not(mask)]
        loss = error ** 2  # + 0.01 * torch.abs(error)
        loss = 10 * torch.mean(loss)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.2g}, Time: {time.time() - st:.2f}')
            st = time.time()

    # Plot reconstructions
    for _ in range(4):
        data, _ = dataloader.get()
        data = data[:, :, :, 13].cuda()
        with torch.no_grad():
            output = model(data)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for i in range(3):
            mins, maxs = ds.ds_min_max[i]
            maxs += 0.2
            axes[0, i].imshow(data[i].cpu().numpy(), vmin=mins, vmax=maxs)
            axes[0, i].set_title(f'Channel {i + 1} Original')
            axes[0, i].axis('off')

            axes[1, i].imshow(output[i].cpu().numpy(), vmin=mins, vmax=maxs)
            axes[1, i].set_title(f'Channel {i + 1} Reconstruction')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.show()

    dataloader.stop()


if __name__ == "__main__":
    train_autoencoder()
