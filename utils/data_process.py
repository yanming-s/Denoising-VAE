import os
import os.path as osp
from PIL import Image
from shutil import rmtree
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import gc


def convert_img_to_tensor(img_dir, save_dir, split_chunk=True, chunk_size=64):
    """
    Convert all JPEG images in a directory to .pt files.
    When split_chunk is True, each .pt file stores up to chunk_size images;
    otherwise, all images are saved in a single .pt file.
    """
    os.makedirs(save_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if split_chunk:
        # Process images in chunks of size chunk_size.
        filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        if not filenames:
            torch.save(torch.empty(0), osp.join(save_dir, "images_0.pt"))
            return
        chunk_idx = 0
        total_chunks = (len(filenames) + chunk_size - 1) // chunk_size
        for i in range(0, len(filenames), chunk_size):
            chunk_files = filenames[i:i+chunk_size]
            chunk_tensors = []
            for filename in chunk_files:
                img_path = osp.join(img_dir, filename)
                image = Image.open(img_path).convert("RGB")
                chunk_tensors.append(transform(image))
            chunk_tensor = torch.stack(chunk_tensors, dim=0)  # shape [chunk_length, C, H, W]
            chunk_file = osp.join(save_dir, f"images_{chunk_idx}.pt")
            torch.save(chunk_tensor, chunk_file)
            chunk_idx += 1
    else:
        # Process all images at once.
        tensors = []
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                img_path = osp.join(img_dir, filename)
                image = Image.open(img_path).convert("RGB")
                tensors.append(transform(image))
        if not tensors:
            torch.save(torch.empty(0), osp.join(save_dir, "images.pt"))
            return
        all_tensors = torch.stack(tensors, dim=0)
        torch.save(all_tensors, osp.join(save_dir, "images.pt"))


def dataset_preprocess(remove_original=False, chunk_size=64):
    """
    Preprocess the dataset by converting all JPEG images to multiple .pt files,
    each chunk having up to 64 images.
    """
    root_data_dir = "split_data/train/"
    root_save_dir = "tensor_data/"
    noise_type = ["defocus_blur", "frost_noise", "gaussian_noise", "jpeg_compression", "speckle_noise"]
    # Process noisy data
    for nt in noise_type:
        img_dir = osp.join(root_data_dir, "noisy", nt)
        save_dir = osp.join(root_save_dir, "noisy", nt)
        convert_img_to_tensor(img_dir, save_dir, split_chunk=True, chunk_size=chunk_size)
        if remove_original:
            rmtree(img_dir, ignore_errors=True)
        print(f"Processed {nt} noisy data.")
    # Process clean data
    img_dir = osp.join(root_data_dir, "clean")
    save_dir = osp.join(root_save_dir, "clean")
    convert_img_to_tensor(img_dir, save_dir, split_chunk=True, chunk_size=chunk_size)
    if remove_original:
        rmtree(img_dir, ignore_errors=True)
    print(f"Processed clean data.")
    # Remove the original data directory
    if remove_original:
        rmtree(root_data_dir, ignore_errors=True)
    print("All datasets have been preprocessed!")


class SimpleChunkCache:
    """
    A simple cache that stores required chunks in memory.
    """
    def __init__(self, max_chunks):
        """
        Initialize a simple cache with maximum number of chunks to store.
        """
        self.max_chunks = max_chunks
        self.cache = {}
    
    def get(self, key):
        """
        Get an item from cache.
        """
        return self.cache.get(key, None)
    
    def put(self, key, value):
        """
        Add an item to cache. If cache is full, clear it first.
        """
        # If we've reached max capacity and this is a new key, clear some space
        if len(self.cache) >= self.max_chunks and key not in self.cache:
            # Clear the entire cache if we're at max capacity
            self.cache.clear()
            # Force garbage collection to free memory
            gc.collect()
        # Add new item
        self.cache[key] = value
    
    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        gc.collect()


class LazyChunkDataset(Dataset):
    """
    A dataset that lazily loads chunks of tensor data from disk as needed.
    """
    def __init__(self, pt_dir, cache_chunks, transform=None):
        """
        Initialize the dataset with a directory of chunk files.
        """
        super().__init__()
        self.pt_dir = pt_dir
        self.transform = transform
        self.cache = SimpleChunkCache(max_chunks=cache_chunks)
        # Count total samples across all chunks
        self.chunks = sorted([f for f in os.listdir(pt_dir) if f.startswith("images_") and f.endswith(".pt")])
        self.chunk_sizes = []
        self.total_samples = 0
        # Get the size of each chunk
        for chunk_file in self.chunks:
            # We'll just load the tensor to get its size, then immediately release it
            chunk_path = osp.join(pt_dir, chunk_file)
            chunk_data = torch.load(chunk_path, map_location='cpu')
            chunk_size = chunk_data.size(0)
            self.chunk_sizes.append(chunk_size)
            self.total_samples += chunk_size
            del chunk_data  # Release memory
        # Create mapping from global index to (chunk_idx, local_idx)
        self.index_map = []
        for chunk_idx, size in enumerate(self.chunk_sizes):
            for local_idx in range(size):
                self.index_map.append((chunk_idx, local_idx))
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range in LazyChunkDataset")
        # Get chunk_idx and local_idx
        chunk_idx, local_idx = self.index_map[idx]
        chunk_file = self.chunks[chunk_idx]
        chunk_key = osp.join(self.pt_dir, chunk_file)
        # Try to get chunk from cache
        chunk_data = self.cache.get(chunk_key)
        if chunk_data is None:
            # Load from disk if not in cache
            chunk_data = torch.load(chunk_key, map_location='cpu')
            self.cache.put(chunk_key, chunk_data)
        # Get the tensor from the chunk
        img_tensor = chunk_data[local_idx]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor
    
    def clear_cache(self):
        """Clear the entire cache."""
        self.cache.clear()


class MultiNoiseDataset(Dataset):
    """
    A dataset that handles multiple noise types, loading data lazily and efficiently.
    """
    def __init__(self, root_tensor_dir, cache_chunks, batch_size=64, transform=None):
        """
        Initialize the dataset with a root directory containing all tensor data.
        """
        super().__init__()
        self.root_tensor_dir = root_tensor_dir
        self.transform = transform
        self.batch_size = batch_size
        # Define noise types
        self.noise_types = ["defocus_blur", "frost_noise", "gaussian_noise", "jpeg_compression", "speckle_noise"]
        # Initialize lazy datasets for each noise type and clean data
        self.clean_dataset = LazyChunkDataset(
            osp.join(root_tensor_dir, "clean"),
            cache_chunks=cache_chunks,
            transform=transform
        )
        print(f"Loaded clean dataset.")
        self.noisy_datasets = {}
        for noise_type in self.noise_types:
            self.noisy_datasets[noise_type] = LazyChunkDataset(
                osp.join(root_tensor_dir, "noisy", noise_type),
                cache_chunks=cache_chunks,
                transform=transform
            )
            print(f"Loaded {noise_type} dataset.")
        # Check that all datasets have the same length
        self.length = len(self.clean_dataset)
        for noise_type, dataset in self.noisy_datasets.items():
            if len(dataset) != self.length:
                raise ValueError(f"Dataset length mismatch: clean ({self.length}) vs {noise_type} ({len(dataset)})")
        # Calculate number of batches
        self.num_batches = self.length // batch_size
        if self.length % batch_size > 0:
            self.num_batches += 1
        # Calculate total number of batches across all noise types
        self.total_batches = self.num_batches * len(self.noise_types)
        # Create shuffled batch indices for the first epoch
        self.reset_indices()
    
    def reset_indices(self):
        """
        Shuffle the batch indices for a new epoch.
        Creates a schedule that randomly selects noise type and batch for each iteration.
        """
        # Create indices for all batches across all noise types
        all_batch_indices = []
        for noise_idx, _ in enumerate(self.noise_types):
            for batch_idx in range(self.num_batches):
                all_batch_indices.append((noise_idx, batch_idx))
        # Shuffle the batch indices
        random.seed(0)  # For reproducibility
        random.shuffle(all_batch_indices)
        self.shuffled_indices = all_batch_indices
        # Reset current batch index
        self.current_batch = 0
    
    def __len__(self):
        return self.total_batches
    
    def __getitem__(self, idx):
        if idx >= self.total_batches:
            raise IndexError("Index out of range in MultiNoiseDataset")
        # Get noise type and batch indices
        noise_idx, batch_idx = self.shuffled_indices[idx]
        noise_type = self.noise_types[noise_idx]
        # Calculate start and end indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.length)
        # Get the noisy and clean tensors for this batch
        noisy_tensors = []
        clean_tensors = []
        for i in range(start_idx, end_idx):
            noisy_tensors.append(self.noisy_datasets[noise_type][i])
            clean_tensors.append(self.clean_dataset[i])
        noisy_batch = torch.stack(noisy_tensors)
        clean_batch = torch.stack(clean_tensors)
        return noisy_batch, clean_batch


def get_multi_noise_dataloader(root_tensor_dir, batch_size=64, num_workers=4, transform=None, cache_chunks=20):
    """
    Create a DataLoader for the MultiNoiseDataset.
    """
    print(f"Loading data from {root_tensor_dir}...")
    dataset = MultiNoiseDataset(
        root_tensor_dir=root_tensor_dir,
        cache_chunks=cache_chunks,
        batch_size=batch_size,
        transform=transform,
    )
    print("Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # returning batches from the dataset
        shuffle=False,  # handle shuffling inside the dataset
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: x[0]  # unwrap the outer batch dimension
    )
    return dataloader
