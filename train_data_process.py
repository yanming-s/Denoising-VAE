import os
import kaggle
from utils.modify_dir import create_directory, delete_directory, rename_directory, copy_jpeg_files, count_jpeg_files, move_directory
from utils.add_noise import process_images, gaussian_noise, speckle_noise, frost, defocus_blur, jpeg_compression
from utils.split_data import split_and_process_dataset
from utils.data_process import dataset_preprocess
from shutil import rmtree
import warnings

warnings.filterwarnings("ignore")


#####################################
###     Author: suryaansh2002     ###
#####################################

# Download the dataset from Kaggle
os.environ["KAGGLE_CONFIG_DIR"] = os.getcwd()  # Use current directory
dataset_identifier = "ifigotin/imagenetmini-1000" 
print("Start downloading dataset...")
kaggle.api.dataset_download_files(dataset_identifier, path="data/", unzip=True)
print("Dataset downloaded successfully in 'data/' folder.")

# Noise the dataset
move_directory('./data/imagenet-mini/train','./')
rename_directory('./train','./imagenet-mini')
create_directory('./gaussian_noise')
create_directory('./speckle_noise')
create_directory('./imagenet-data')
count_jpeg_files('./imagenet-mini')
copy_jpeg_files('./imagenet-mini','./imagenet-data')
delete_directory('./imagenet-mini')
delete_directory('./data')
severity_levels = 5 # Number of severity levels for noise, can be [1, 2, 3, 4, 5]
process_images('./imagenet-data', './gaussian_noise', gaussian_noise, severity_levels)
process_images('./imagenet-data', './speckle_noise', speckle_noise, severity_levels)
process_images('./imagenet-data', './frost_noise', frost, severity_levels)
process_images('./imagenet-data', './defocus_blur', defocus_blur, severity_levels)
process_images('./imagenet-data', './jpeg_compression', jpeg_compression, severity_levels)
rename_directory('./imagenet-data','./clean')
create_directory('./noisy')
noisy_directories =['./gaussian_noise','./speckle_noise', './frost_noise', './defocus_blur', './jpeg_compression']
for noisy_dir in noisy_directories:
    rename_directory(noisy_dir+f'/{severity_levels}', noisy_dir+f'/{noisy_dir.split("/")[-1]}')
for noisy_dir in noisy_directories:
    move_directory(noisy_dir+f'/{noisy_dir.split("/")[-1]}', './noisy')
create_directory('./final_data')
move_directory('./clean', './final_data')
move_directory('./noisy', './final_data')
split_and_process_dataset('./final_data','./split_data')
rmtree('./final_data')
for noisy_dir in noisy_directories:
    rmtree(noisy_dir)

# Convert the dataset to tensors
print("Converting images to tensors...")
dataset_preprocess()
print("Conversion complete.")
