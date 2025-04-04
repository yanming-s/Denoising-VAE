import shutil
import os
from tqdm import tqdm

def print_dir_structure(root_dir, indent=""):
    """
    Recursively prints the directory structure.
    - For each directory, it prints its name, a preview of the first few files,
      and the total number of files.
    - If a directory has more than 20 subdirectories, it prints details for
      the first 20 and then prints a summary line with the remaining count.
    
    Args:
        root_dir (str): The path of the directory to print.
        indent (str): String used for indentation (for recursive calls).
    """
    # Print the directory name
    print(f"{indent}{os.path.basename(root_dir)}/")
    
    # Get all entries in the directory
    try:
        entries = os.listdir(root_dir)
    except PermissionError:
        print(f"{indent}   [Permission Denied]")
        return
    
    # Separate files and directories
    files = [f for f in entries if os.path.isfile(os.path.join(root_dir, f))]
    sub_dirs = [d for d in entries if os.path.isdir(os.path.join(root_dir, d))]
    
    # Print file details if present
    if files:
        # Show first 5 file names as a preview
        file_preview = ", ".join(files[:5])
        if len(files) > 5:
            file_preview += "..."
        print(f"{indent}   Files: {file_preview}")
        print(f"{indent}   [Total Files: {len(files)}]")
    
    # Process subdirectories
    if sub_dirs:
        # If there are more than 20 subdirectories, only show details for the first 20
        if len(sub_dirs) > 20:
            for d in sub_dirs[:20]:
                print_dir_structure(os.path.join(root_dir, d), indent + "   ")
            print(f"{indent}   ... and {len(sub_dirs) - 20} more directories")
        else:
            for d in sub_dirs:
                print_dir_structure(os.path.join(root_dir, d), indent + "   ")

def list_all_dirs(root_dir, max_level, current_level=1):
    """
    Recursively list all directories within the given root directory up to max_level.
    
    Args:
        root_dir (str): The starting directory.
        max_level (int): Maximum level to descend. Level 1 means only direct subdirectories.
        current_level (int): Current level of recursion (used internally).
    
    Returns:
        List[str]: A list of paths to all found directories up to the specified level.
    """
    directories = []
    # Stop recursion if the current level exceeds max_level
    if current_level > max_level:
        return directories

    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if os.path.isdir(path):
            directories.append(path)
            # Continue only if we haven't reached the maximum level
            directories.extend(list_all_dirs(path, max_level, current_level + 1))
    return directories


def create_directory(directory_path):
    """Creates a directory if it does not exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully.")
    except Exception as e:
        print(f"Error creating directory '{directory_path}': {e}")

def rename_directory(old_name, new_name):
    """Renames a directory."""
    try:
        os.rename(old_name, new_name)
        print(f"Directory '{old_name}' renamed to '{new_name}'.")
    except Exception as e:
        print(f"Error renaming directory '{old_name}': {e}")

def copy_directory(src, dest):
    """Copies a directory from src to dest."""
    try:
        shutil.copytree(src, dest)
        print(f"Directory '{src}' copied to '{dest}'.")
    except Exception as e:
        print(f"Error copying directory '{src}': {e}")

def delete_directory(directory_path):
    """Deletes a directory and all its contents."""
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting directory '{directory_path}': {e}")

def copy_jpeg_files(src, dest):
    """Recursively finds all .JPEG files in src and copies them to dest."""
    try:
        if not os.path.exists(dest):
            os.makedirs(dest)
        
        for root, _, files in os.walk(src):
            for file in files:
                if file.lower().endswith(".jpeg"):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest, file)
                    shutil.copy2(src_file, dest_file)
    except Exception as e:
        print(f"Error copying JPEG files: {e}")


def count_jpeg_files(directory):
    count = 0
    for root, _, files in os.walk(directory):
        jpeg_files = [f for f in files if f.lower().endswith('.jpeg')]
        count += len(jpeg_files)

    print(f"Total .jpeg files: {count}")

def move_directory(src_dir, dest_parent_dir):
    """
    Moves the entire src_dir under dest_parent_dir.
    
    Example:
    move_directory('folderA', 'target/')
    Result: target/folderA/
    """
    # Ensure the destination parent exists
    os.makedirs(dest_parent_dir, exist_ok=True)

    # Get the folder name to preserve structure
    folder_name = os.path.basename(os.path.normpath(src_dir))
    dest_path = os.path.join(dest_parent_dir, folder_name)

    # Move the directory
    shutil.move(src_dir, dest_path)

    print(f"Moved '{src_dir}' to '{dest_path}'")




def organize_images_by_class(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Loop through each image in the source directory
    for filename in tqdm(os.listdir(src_dir)):
        if filename.lower().endswith('.jpeg'):
            class_name = filename.split('_')[0]
            class_dir = os.path.join(dst_dir, class_name)

            # Create subdirectory if it doesn't exist
            os.makedirs(class_dir, exist_ok=True)

            # Copy image to its class subdirectory
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(class_dir, filename)

            shutil.copy2(src_path, dst_path)
