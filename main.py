import os
import os.path as osp
import torch
from pytorch_lightning.trainer import Trainer
import wandb
import time

from utils.data_process import get_multi_noise_dataloader
from models.vae import VAE, VAE_Module


def main(model_name = "vae", reload_model=False):
    # Load the dataset
    root_tensor_dir = "tensor_data/"
    train_dataloader = get_multi_noise_dataloader(root_tensor_dir, batch_size=64)
    
    # Initialize the model
    if model_name == "vae":
        model_args = {
            "img_size": 224,
            "patch_size": 16,
            "in_channels": 3,
            "latent_dim": 512,
            "embed_dim": 768,
            "encoder_depth": 12,
            "decoder_depth": 12,
            "num_heads": 12, 
            "mlp_dim": 768*4,
            "dropout": 0.1
        }
        model = VAE(**model_args)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model_path = f"{model_name}.pth"
    if reload_model and osp.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # Initialize wandb
    date = time.strftime("%Y-%m-%d")
    timestamp = time.strftime("%H-%M-%S")
    save_dir = f"logs/{date}/{model_name}-{timestamp}"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    wandb.init(
        project="CS5340-VAE-Denoising",
        name=f"{model_name}_multi_noise",
        dir=save_dir,
        config=model_args,
        mode="online"
    )
    
    # Initialize lightning modules
    if model_name == "vae":
        train_module = VAE_Module(
            model=model,
            train_loader=train_dataloader
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Set up the trainer
    gpu = torch.cuda.is_available()
    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu" if gpu else "cpu",
        devices=[0] if gpu else 1,
        precision="16-mixed",
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
    save_path = osp.join(save_dir, f"{model_name}.pth")
    if osp.exists(save_path):
        os.remove(save_path)
    torch.save(model.state_dict(), save_path)
    print(f"\n>>> Model saved at {save_path}\n")


if __name__ == "__main__":
    main("vae")
