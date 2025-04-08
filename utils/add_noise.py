# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as trn
import torch.utils.data as data
from PIL import Image
from skimage.util import random_noise
from skimage.filters import gaussian
import cv2
from scipy.ndimage import map_coordinates
from io import BytesIO


# Image formats supported
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# Function to check if a file is an image
def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

# Function to find image classes
def find_classes(root_dir):
    class_names = set()
    for img_name in os.listdir(root_dir):
        if is_image_file(img_name):
            class_name = img_name.split('_')[0]
            class_names.add(class_name)
    classes = sorted(class_names)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    return classes, class_to_idx

# Function to create dataset of image paths and labels
def make_dataset(root_dir, class_to_idx):
    images = []
    for img_name in sorted(os.listdir(root_dir)):
        if is_image_file(img_name):
            img_path = os.path.join(root_dir, img_name)
            class_name = img_name.split('_')[0]  # Extract class from filename
            if class_name in class_to_idx:
                images.append((img_path, class_name))  # Store path and class
    return images


# Function to load an image using PIL
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

# Dataset class to apply distortions
class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, severity, save_root, transform=None, loader=pil_loader):
        self.root = root
        self.method = method
        self.severity = severity
        self.save_root = save_root
        self.transform = transform
        self.loader = loader

        classes, class_to_idx = find_classes(root)
        self.imgs = make_dataset(root, class_to_idx)
        self.classes = classes

        if len(self.imgs) == 0:
            raise RuntimeError(f"Found 0 images in: {root}")

    def __getitem__(self, index):
        img_path, class_name = self.imgs[index]
        img = self.loader(img_path)  # Load the image

        if self.transform:
            img = self.transform(img)  # Apply any transformations (e.g., resize)

        img = self.method(img, self.severity)  # Apply noise distortion

        # Define the save path
        os.makedirs(self.save_root, exist_ok=True)
        # Optional: prefix class name if not already part of filename
        filename = os.path.basename(img_path)
        save_path = os.path.join(self.save_root, filename)
        img.save(save_path, quality=85, optimize=True)  # Save processed image

        return 0  # No need to return data for DataLoader

    def __len__(self):
        return len(self.imgs)

# ---------------------- Distortion Functions ---------------------- #

def gaussian_noise(img, severity=1):
    """
    Applies Gaussian noise to an image.
    """
    severity_levels = [0.04, 0.08, 0.12, 0.15, 0.18]
    c = severity_levels[severity - 1]

    img = np.array(img) / 255.0  # Normalize
    noisy_img = np.clip(img + np.random.normal(scale=c, size=img.shape), 0, 1) * 255
    return Image.fromarray(np.uint8(noisy_img))

def speckle_noise(img, severity=1):
    """
    Applies Speckle noise to an image.
    """
    severity_levels = [0.15, 0.2, 0.25, 0.3, 0.35]
    c = severity_levels[severity - 1]

    img = np.array(img) / 255.0  # Normalize
    noisy_img = np.clip(img + img * np.random.normal(scale=c, size=img.shape), 0, 1) * 255
    return Image.fromarray(np.uint8(noisy_img))



def frost(img, severity=1):
    """
    Applies a 'frosted' effect to a PIL Image of arbitrary size.
    Returns a PIL Image.
    """
    # Convert the PIL image to a NumPy array
    img_np = np.array(img, dtype=np.float32)

    # Blending coefficients
    c = [(1, 0.3), (0.9, 0.4), (0.8, 0.45), (0.75, 0.5), (0.7, 0.55)][severity - 1]

    # Load a random frost image (BGR)
    idx = np.random.randint(5)
    filename = [
        './images/frost1.png', './images/frost2.png', './images/frost3.png',
        './images/frost4.jpg', './images/frost5.jpg', './images/frost6.jpg'
    ][idx]
    frost = cv2.imread(filename)

    # If the file was not found or frost is None, handle it gracefully
    if frost is None:
        # As a fallback, just return the original
        return img

    # Convert BGR -> RGB
    frost = frost[..., [2, 1, 0]]

    # Resize the frost image to match the input's height & width
    h, w = img_np.shape[:2]
    frost_resized = cv2.resize(frost, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Blend them: out = c[0]*x + c[1]*frost
    out_np = c[0] * img_np + c[1] * frost_resized
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)

    # Convert the blended NumPy array back to a PIL image
    out_pil = Image.fromarray(out_np)
    return out_pil


def defocus_blur(img, severity=1):
    """
    Applies a defocus blur to a PIL Image.
    The function handles arbitrary image sizes and returns a PIL image.
    """
    # Blending parameters for defocus blur
    c = [(0.5, 0.6), (1, 0.1), (1.5, 0.1), (2.5, 0.01), (3, 0.1)][severity - 1]
    
    # Convert input PIL image to a normalized NumPy array (values in [0,1])
    x = np.array(img).astype(np.float32) / 255.0
    
    # 'disk' is assumed to be defined elsewhere in your code.
    kernel = disk(radius=c[0], alias_blur=c[1])
    
    # Apply the kernel on each channel separately
    channels = []
    for d in range(3):
        ch = cv2.filter2D(x[:, :, d], -1, kernel)
        channels.append(ch)
    # Reassemble the channels back into an image array
    out = np.stack(channels, axis=2)
    
    # Clip and convert back to 8-bit values
    out = np.clip(out, 0, 1) * 255
    out = np.uint8(out)
    
    # Return a PIL image
    return Image.fromarray(out)



def jpeg_compression(x, severity=1):
    """
    Simulates JPEG compression on a PIL Image by saving and reopening it
    with a specified JPEG quality.
    
    Parameters:
        x (PIL.Image): The input image.
        severity (int): An integer from 1 to 5 determining the compression level.
                        Lower quality (higher severity) introduces more compression artifacts.
                        
    Returns:
        PIL.Image: The JPEG compressed image.
    """
    # Choose the JPEG quality based on severity. Lower values mean stronger compression.
    quality = [65, 58, 50, 40, 25][severity - 1]

    # Create an in-memory bytes buffer.
    output = BytesIO()
    
    # Save the image to the buffer in JPEG format using the selected quality.
    x.save(output, 'JPEG', quality=quality)
    
    # Rewind the buffer's file pointer to the beginning.
    output.seek(0)
    
    # Open the image from the buffer, which now includes JPEG compression artifacts.
    compressed_image = Image.open(output)
    
    return compressed_image


# ---------------------- Processing Function ---------------------- #

def process_images(input_folder, output_folder, method, severity_levels=3, resize=False):
    """
    Applies a given noise method to all images in the input folder
    and saves the output to the specified output folder.
    """
    for severity in range(severity_levels, severity_levels + 1):
        print(f"Applying {method.__name__} with severity {severity}...")

        dataset = DistortImageFolder(
            root=input_folder,
            method=method,
            severity=severity,
            save_root=os.path.join(output_folder, str(severity)),  # Save under severity level
            transform = trn.Compose([trn.Resize((64, 64))]) if resize else None,
            # transform=None
        )

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

        for _ in dataloader:
            pass  # Just to trigger processing

