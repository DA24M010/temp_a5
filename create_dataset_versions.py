import os
import shutil
import tarfile
import random
import subprocess
from glob import glob
from tqdm import tqdm
import urllib.request
from cifar10_extraction import extract_images  # You need a separate script to convert CIFAR batch files to images

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TAR = "cifar-10.tar.gz"
EXTRACTED_DIR = "cifar-10-batches-py"
IMAGE_DATASET = "cifar10_images"
DATASET_DIR = "dataset"
PARTITION_SIZE = 20000
GIT_IGNORE = ".gitignore"

# Step 1: Download CIFAR-10 if not already present
if not os.path.exists(CIFAR_TAR):
    print("Downloading CIFAR-10 dataset...")
    urllib.request.urlretrieve(CIFAR_URL, CIFAR_TAR)

# Step 2: Extract CIFAR-10 if not already extracted
if not os.path.exists(EXTRACTED_DIR):
    print("Extracting CIFAR-10 dataset...")
    with tarfile.open(CIFAR_TAR, "r:gz") as tar:
        tar.extractall()

# Remove tar file
if os.path.exists(CIFAR_TAR):
    os.remove(CIFAR_TAR)

# Step 3: Convert CIFAR-10 batches to images
if not os.path.exists(IMAGE_DATASET):
    os.makedirs(IMAGE_DATASET)
    extract_images(EXTRACTED_DIR, IMAGE_DATASET)

# Remove cifar-10-batches-py folder
if os.path.exists(EXTRACTED_DIR):
    shutil.rmtree(EXTRACTED_DIR)

# Step 4: Initialize Git & DVC
if not os.path.exists(".git"):
    subprocess.run(["git", "init"])
    subprocess.run(["dvc", "init"])
    subprocess.run(["touch", GIT_IGNORE])

# Step 5: Add CIFAR-10 folder to `.gitignore`
with open(GIT_IGNORE, "a") as f:
    f.write(IMAGE_DATASET + "\n")

subprocess.run(["git", "commit", "-m", f"Initialized repo and added .gitignore"])

all_images = glob(os.path.join(IMAGE_DATASET, "*", "*.png"))
random.shuffle(all_images)
if len(all_images) != 60000:
    raise ValueError("Dataset extraction failed. Found incorrect number of images!")

for version in range(1, 4):
    print(f"Creating dataset partition v{version}...")

    if os.path.exists(DATASET_DIR):
        for item in os.listdir(DATASET_DIR):
            item_path = os.path.join(DATASET_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove subfolders
            else:
                os.remove(item_path)  # Remove individual files
    else:
        os.makedirs(DATASET_DIR)

    partition = all_images[(version-1)*PARTITION_SIZE : version*PARTITION_SIZE]
    for img in tqdm(partition):
        class_folder = os.path.basename(os.path.dirname(img))
        dest_folder = os.path.join(DATASET_DIR, class_folder)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(img, dest_folder)

    # Step 7: Track dataset in DVC and Git
    subprocess.run(["dvc", "add", DATASET_DIR])
    subprocess.run(["git", "add", GIT_IGNORE, f"{DATASET_DIR}.dvc"])
    subprocess.run(["git", "commit", "-m", f"Added dataset partition v{version}"])
    subprocess.run(["git", "tag", f"v{version}", "-m", f"Version {version} dataset partition"])

# Step 8: Push to remote DVC storage (if configured)
# subprocess.run(["dvc", "push"])


