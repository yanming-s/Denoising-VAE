import os
import shutil
import random

def split_and_process_dataset(original_data_dir, formatted_data_dir, split_ratios={"train": 0.8, "validation": 0.2}):
    """Splits dataset into train and validation sets and organizes them into separate directories."""
    # Ensure reproducibility
    random.seed(42)
    
    # Get all filenames from the clean folder (since they are the same in noisy folders)
    clean_dir = os.path.join(original_data_dir, "clean")
    all_filenames = sorted(os.listdir(clean_dir))  # Sort to maintain consistency

    # Shuffle and split
    random.shuffle(all_filenames)
    total_files = len(all_filenames)
    
    train_count = int(total_files * split_ratios["train"])
    
    train_files = set(all_filenames[:train_count])
    val_files = set(all_filenames[train_count:])

    def process_directory(src_dir, dest_dir):
        """Recursively processes the directory structure and moves files to train/val splits."""
        for root, _, files in os.walk(src_dir):
            relative_path = os.path.relpath(root, original_data_dir)
            for category, file_set in [("train", train_files), ("validation", val_files)]:
                category_dest_dir = os.path.join(dest_dir, category, relative_path)
                os.makedirs(category_dest_dir, exist_ok=True)

                for file in files:
                    if file in file_set:
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(category_dest_dir, file)
                        shutil.copy2(src_file, dest_file)  # Preserve metadata
    
    # Process clean and noisy data
    process_directory(os.path.join(original_data_dir, "clean"), formatted_data_dir)
    process_directory(os.path.join(original_data_dir, "noisy"), formatted_data_dir)

    print("Dataset successfully split into train and validation sets.")
