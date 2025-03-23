# import yaml
# import os
# import pandas as pd
# from glob import glob

# CONFIG_PATH = "params.yaml"
# DATASET_PATH = "dataset/"

# # Load YAML configuration
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# dataset_version = config["dataset"]["version"]

# # Pull dataset from DVC
# os.system(f"git checkout {dataset_version}") 
# os.system("dvc checkout") # Checkout the correct dataset version

# # Load images into a dataframe
# image_paths = glob(f"{DATASET_PATH}/**/*.png", recursive=True) + glob(f"{DATASET_PATH}/**/*.png", recursive=True)
# df = pd.DataFrame({"image_path": image_paths, "label": [path.split("/")[-2] for path in image_paths]})
# print(df)
# # Save as a pickle file
# df.to_pickle("dataset.pkl")

# print(f"✅ Pulled dataset version: {dataset_version} and saved as dataframe.")

# os.system("git add .gitignore dvc.lock dvc.yaml") 



import yaml
import os
import pandas as pd
from glob import glob


CONFIG_PATH = "params.yaml"
DATASET_PATH = "dataset/"  # Update if your datasets are stored elsewhere

# Load YAML configuration
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if os.path.exists("dataset.pkl"):
    os.remove("dataset.pkl")
    print("Deleted existing dataset.pkl")

# Parse dataset versions from params.yaml
dataset_param = config["dataset"]["version"]  # Example: "v1", "v1+v2", "v1+v2+v3"
dataset_versions = dataset_param.split("+")  # Split into a list: ["v1", "v2", "v3"]

all_data = []

for version in dataset_versions:
    print(f"Checking out dataset version: {version}")

    # Switch to the dataset version
    os.system(f"git checkout {version}")  
    os.system("dvc checkout")  

    # Collect image paths
    image_paths = glob(f"{DATASET_PATH}/**/*.png", recursive=True) + glob(f"{DATASET_PATH}/**/*.jpg", recursive=True)
    df = pd.DataFrame({"image_path": image_paths, "label": [path.split("/")[-2] for path in image_paths]})

    if not image_paths:
        print(f"❌ No images found for {version}. Check dataset structure!")
        continue  # Skip this version if no images are found

    image_data = []
    
    for img_path in image_paths:
        # Read image, convert to bytes, and store it
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
        
        label = img_path.split("/")[-2]  # Assuming folder names are labels
        image_data.append({"image_bytes": img_bytes, "label": label})
    
    df = pd.DataFrame(image_data)
    all_data.append(df)

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"Images stored in dataframe shape : {final_df.shape}")

    final_df.to_pickle("dataset.pkl")  # Save for later processing
    print(f"✅ Dataset {dataset_versions} merged and saved as dataset.pkl.")
else:
    print("❌ No valid datasets found. Check dataset paths.")

# Stage changes for tracking
os.system("git add .gitignore dvc.lock dvc.yaml")  
