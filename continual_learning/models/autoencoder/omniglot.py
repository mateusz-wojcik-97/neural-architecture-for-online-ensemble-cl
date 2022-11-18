from typing import Tuple, Any, Optional

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer, Callback
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision.transforms import Compose

from settings import DATASETS_DIR


class OmniglotAutoencoderModel(nn.Module):
    def __init__(self, input_size: int = 28, encoder_size: int = 28):
        super().__init__()
        self.input_size = input_size
        self.encoder_size = encoder_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, padding='same'),
            nn.ReLU(),
            nn.Flatten()
        )
        self.encoder_output_tanh = nn.Sequential(
            nn.Linear(12544, 128),
            nn.ReLU(),
            nn.Linear(128, self.encoder_size),
            nn.Tanh()
        )
        self.encoder_output_relu = nn.Sequential(
            nn.Linear(12544, 128),
            nn.ReLU(),
            nn.Linear(128, self.encoder_size),
            nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(self.encoder_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_size * self.input_size * 16),
            nn.ReLU(),
            nn.Unflatten(1, (16, self.input_size, self.input_size)),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.ZeroPad2d(padding=(0, 1, 0, 1)),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, padding=2),
            nn.Sigmoid()
        )

    def encode(self, x):
        cnn_encoding = self.encoder(x)
        tanh_encoding = self.encoder_output_tanh(cnn_encoding)
        relu_encoding = self.encoder_output_relu(cnn_encoding)
        return tanh_encoding, relu_encoding

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        tanh_encoding, relu_encoding = self.encode(x)
        joint_encoding = tanh_encoding + tanh_encoding + relu_encoding * torch.empty(relu_encoding.size()).normal_(mean=0, std=1)
        reconstruction = self.decode(joint_encoding)
        return reconstruction, joint_encoding, tanh_encoding, relu_encoding


class OmniglotAutoencoder(LightningModule):
    def __init__(self, input_size: int, encoder_size: int, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = OmniglotAutoencoderModel(input_size=input_size, encoder_size=encoder_size).to(torch.device('cuda'))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model.forward(x)

    def loss(self, batch, reconstruction, tanh_encoding, relu_encoding):
        mse_loss = F.mse_loss(reconstruction, batch)
        kl_loss = -0.5 * (1.0 + torch.log(relu_encoding ** 2) - tanh_encoding ** 2 - relu_encoding ** 2)
        kl_loss_mean = kl_loss.mean()
        return mse_loss + 0.001 * kl_loss_mean, mse_loss, kl_loss_mean

    def training_step(self, batch, batch_nb):
        x, y = batch
        reconstruction, joint_encoding, tanh_encoding, relu_encoding = self.forward(x)
        loss, mse_loss, kl_loss = self.loss(x, reconstruction, tanh_encoding, relu_encoding + 1e-10)

        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/reconstruction_loss", mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        reconstruction, joint_encoding, tanh_encoding, relu_encoding = self.forward(x)
        loss, mse_loss, kl_loss = self.loss(x, reconstruction, tanh_encoding, relu_encoding + 1e-10)

        self.log('val/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/reconstruction_loss", mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/kl_loss", kl_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss


class ThresholdStopping(Callback):
    def __init__(self, metric: str, threshold: float):
        super().__init__()
        self.metric = metric
        self.threshold = threshold

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics

        if logs:
            current = logs[self.metric].squeeze()
            loss_satisfied = current <= self.threshold

            trainer.should_stop = trainer.should_stop or loss_satisfied


if __name__ == '__main__':
    dataset_dir = DATASETS_DIR / 'omniglot'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset = torchvision.datasets.Omniglot(
        root=dataset_dir,
        download=True,
        transform=Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((28, 28))
        ])
    )
    data_loader = DataLoader(dataset, batch_size=48)

    model = OmniglotAutoencoderModel(encoder_size=28)
    autoencoder = OmniglotAutoencoder(input_size=28, encoder_size=28, learning_rate=0.001)

    trainer = Trainer(
        max_epochs=1,
        progress_bar_refresh_rate=10,
        enable_progress_bar=True,
        enable_checkpointing=False,
        checkpoint_callback=False,
        logger=True,
        weights_summary=None,
    )

    # Loss threshold 0.025
    trainer.fit(autoencoder, data_loader)

    # T-SNE visualization
    for batch in data_loader:
        reconstruction = model.forward(batch)

    # for item, label in data_loader:
        # print(item.size(), label.size())
        # res = model.forward(item)
        # print(res.size())
