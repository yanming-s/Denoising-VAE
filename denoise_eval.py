import os
import os.path as osp
import time
import torch
import matplotlib.pyplot as plt

from models.vae import VAE


def denormalize(tensor):
    """
    Denormalize a tensor image to [0, 1] range.
    Args:
        tensor (torch.Tensor): The input tensor image.
    Returns:
        torch.Tensor: The denormalized tensor image.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    tensor = tensor * std + mean
    tensor = torch.clip(tensor, 0, 1)  # Ensure values are in [0, 1]
    return tensor


def main(index=0, ckpt_path="checkpoints/vae.pth"):
    """
    Denoising demo for the VAE model.
    Args:
        index (int): Index of the image to denoise.
        ckpt_path (str): Path to the model checkpoint.
    """
    # Load batch samples
    clean_batch = torch.load("tensor_data/clean/images_0.pt", weights_only=True)
    defocus_batch = torch.load("tensor_data/noisy/defocus_blur/images_0.pt", weights_only=True)
    frost_batch = torch.load("tensor_data/noisy/frost_noise/images_0.pt", weights_only=True)
    gaussian_batch = torch.load("tensor_data/noisy/gaussian_noise/images_0.pt", weights_only=True)
    jpeg_batch = torch.load("tensor_data/noisy/jpeg_compression/images_0.pt", weights_only=True)
    speckle_batch = torch.load("tensor_data/noisy/speckle_noise/images_0.pt", weights_only=True)
    # Create image saved directory
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"imgs/sample-{timestamp}"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    # Draw the images
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(denormalize(clean_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Clean Image")
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(denormalize(defocus_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Defocus Blur")
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(denormalize(frost_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Frost Noise")
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(denormalize(gaussian_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Gaussian Noise")
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(denormalize(jpeg_batch[index].permute(1, 2, 0)).numpy())
    plt.title("JPEG Compression")
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(denormalize(speckle_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Speckle Noise")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f"sample-{index}.png"))
    # Load the model
    model = VAE()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model.to(device)
    # Denoise the images
    sample_batch = gaussian_batch.to(device)
    recong_batch, _, _ = model(sample_batch)
    recong_batch = recong_batch.detach().cpu()
    sample_batch = sample_batch.detach().cpu()
    # Draw the images
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(denormalize(sample_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Clean Image", fontsize=25)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(denormalize(recong_batch[index].permute(1, 2, 0)).numpy())
    plt.title("Reconstructed Image", fontsize=25)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(osp.join(save_dir, f"denoise-{index}.png"))

if __name__ == "__main__":
    index = 0
    ckpt_path = "checkpoints/vae.pth"
    main(index=index, ckpt_path=ckpt_path)
