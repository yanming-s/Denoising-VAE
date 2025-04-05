import os
import os.path as osp
import torch
from pytorch_lightning.trainer import Trainer
import wandb
import time

from utils.data_process import get_multi_noise_dataloader
from models.dvae import DVAE, DVAE_Module


def main(reload_model=False):
    # Load the dataset
    root_tensor_dir = "tensor_data/"
    train_dataloader = get_multi_noise_dataloader(root_tensor_dir, batch_size=64)
    
    # Initialize the model
    model_args = {
        "image_size": 224,
        "n_channels": 3,
        "latent_dim": 512,
        "bilinear": True
    }
    model = DVAE(**model_args)
    model_path = "dvae.pth"
    if reload_model and osp.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # Initialize wandb
    date = time.strftime("%Y-%m-%d")
    timestamp = time.strftime("%H-%M-%S")
    save_dir = f"logs/{date}/dvae-{timestamp}"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    wandb.init(
        project="DVAE",
        name=f"denoising-vae-unet",
        dir=save_dir,
        config=model_args,
        mode="online"
    )
    
    # Initialize lightning modules
    train_module = DVAE_Module(
        model=model,
        train_loader=train_dataloader
    )

    # Set up the trainer
    gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if gpu else "cpu",
        devices=[0] if gpu else 1,
        callbacks=[],
        enable_progress_bar=False,
        log_every_n_steps=1,
        logger=[],
        enable_checkpointing=False
    )

    # Train the model
    trainer.fit(train_module, train_dataloader)

    # Save the model
    save_dir = "checkpoints"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, "dvae.pth")
    if osp.exists(save_path):
        os.remove(save_path)
    torch.save(model.state_dict(), save_path)
    print(f"\n>>> Model saved at {save_path}\n")


if __name__ == "__main__":
    main()
