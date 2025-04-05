import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import os
import os.path as osp
from time import time
import wandb


class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net
    (Conv -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 dimensions if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class Reparameterize(nn.Module):
    """
    Reparameterization trick for VAE
    """
    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class DVAE(nn.Module):
    """
    Denoising Variational Autoencoder based on U-Net architecture
    """
    def __init__(self, image_size=224, n_channels=3, latent_dim=512, bilinear=True):
        super(DVAE, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.latent_dim = latent_dim
        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # Calculate the flattened feature map size based on input size
        feature_size = image_size // (2 ** 4)
        self.flattened_size = (1024 // factor) * feature_size * feature_size
        # Latent encoding (mu and logvar)
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.reparameterize = Reparameterize()
        # Latent to feature map
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        # Decoder (expanding path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)
    def encode(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Flatten the feature map
        x5_flat = x5.view(x5.size(0), -1)
        # Get latent parameters
        mu = self.fc_mu(x5_flat)
        logvar = self.fc_logvar(x5_flat)
        return x1, x2, x3, x4, x5, mu, logvar
    def decode(self, z, x1, x2, x3, x4, x5):
        # Reconstruct feature map
        z_decode = self.fc_decode(z)
        z_unflatten = z_decode.view(z.size(0), x5.size(1), x5.size(2), x5.size(3))
        # Decoder path with skip connections
        x = self.up1(z_unflatten, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    def forward(self, x):
        # Encode
        x1, x2, x3, x4, x5, mu, logvar = self.encode(x)
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        # Decode
        recon_x = self.decode(z, x1, x2, x3, x4, x5)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, log_var, kld_weight=0.0):
    """
    VAE loss function = Reconstruction loss + KL divergence
    """
    batch_size = x.size(0)
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    # Total loss
    loss = recon_loss + kld_weight * kld_loss
    return loss, recon_loss, kld_loss


class DVAE_Module(LightningModule):
    """
    PyTorch Lightning module for VAE Training
    """
    def __init__(self, model, train_loader, lr=1e-4, scheduler="cosine", save_ckpt=True,
                 save_every_epoch=5, save_dir="checkpoints", max_grad_norm=1.0):
        super().__init__()
        # Module setup
        self.model = model
        self.train_loader = train_loader
        # Learning settings
        self.lr = lr
        self.kl_weight = 0.0
        self.max_grad_norm = max_grad_norm
        if scheduler not in ["cosine", "plateau"]:
            raise ValueError(f"Invalid scheduler: {scheduler}. Must be 'cosine' or 'plateau'.")
        self.scheduler = scheduler
        # Save settings
        self.save_dir = save_dir
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        self.save_ckpt = save_ckpt
        self.save_every_epoch = save_every_epoch
        # Running loss
        self.train_step_outputs = []
        # Running time
        self.epoch_start_time = None
    
    def train_dataloader(self):
        return self.train_loader
    
    def training_step(self, batch, _):
        noisy_x, clean_x = batch
        recon_x, mu, log_var = self.model(noisy_x)
        loss, _, _ = vae_loss(recon_x, clean_x, mu, log_var, self.kl_weight)
        self.train_step_outputs.append(loss.detach())
        return loss
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time()

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.print(
            f"Epoch {self.current_epoch + 1} / {self.trainer.max_epochs}"
            f" - Train Loss: {avg_loss.item():.6f}"
            f" - Time: {(time() - self.epoch_start_time) / 60:.2f} mins"
        )
        wandb.log({"train_loss": avg_loss.item()})
        self.log("train_loss", avg_loss.item())
        self.train_step_outputs.clear()
        # Update KL weight and max grad norm
        if self.current_epoch >= 5:
            self.kl_weight = min(0.5, self.current_epoch / 80)
        # Save model
        if self.save_ckpt and (self.current_epoch + 1) % self.save_every_epoch == 0:
            save_path = osp.join(self.save_dir, f"epoch_{self.current_epoch + 1}.pth")
            torch.save(self.model.state_dict(), save_path)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        if self.scheduler == "cosine":
            # Cosine annealing with warm restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
                "gradient_clip_val": self.max_grad_norm
            }
        else:
            # Reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=False,
                min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
                "gradient_clip_val": self.max_grad_norm
            }
