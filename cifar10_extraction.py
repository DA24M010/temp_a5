import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# CIFAR-10 batch file metadata
CIFAR_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def unpickle(file):
    """Load CIFAR-10 dataset batch files."""
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def extract_images(cifar_folder, output_folder):
    """
    Extracts images from CIFAR-10 batch files and saves them as PNGs in class folders.

    :param cifar_folder: The folder containing extracted CIFAR-10 batch files.
    :param output_folder: The folder where extracted images will be stored.
    """
    os.makedirs(output_folder, exist_ok=True)

    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    image_id = 0  # Unique ID for each image

    for batch in batch_files:
        batch_path = os.path.join(cifar_folder, batch)
        
        if not os.path.exists(batch_path):
            print(f"Batch file {batch_path} not found! Skipping...")
            continue

        print(f"Processing {batch}...")
        data_dict = unpickle(batch_path)
        images = data_dict[b"data"]
        labels = data_dict[b"labels"]

        # Reshape images (CIFAR-10 stores them as 1D arrays)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        for img, label in tqdm(zip(images, labels), total=len(images)):
            class_name = CIFAR_LABELS[label]
            class_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)

            # Save image as PNG
            img_path = os.path.join(class_folder, f"{image_id}.png")
            Image.fromarray(img).save(img_path)
            image_id += 1

    print(f"✅ Extraction complete! Images saved in '{output_folder}/'.")

