import yaml
import pandas as pd
import numpy as np
import os

CONFIG_PATH = "params.yaml"

# Load YAML config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if os.path.exists("dataset_splits.pkl"):
    os.remove("dataset_splits.pkl")
    print("Deleted existing dataset_splits.pkl")

train_ratio = config["split"]["train"]
val_ratio = config["split"]["val"]
test_ratio = config["split"]["test"]
seed = config["seed"]

# Load dataset
df = pd.read_pickle("dataset.pkl")

# Shuffle dataset
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# Compute split sizes
train_end = int(len(df) * train_ratio)
val_end = train_end + int(len(df) * val_ratio)

# Split into train, val, test
df_train = df.iloc[:train_end]
print(df_train.shape)
df_val = df.iloc[train_end:val_end]
print(df_val.shape)
df_test = df.iloc[val_end:]
print(df_test.shape)

# Save splits
df_splits = {"train": df_train, "val": df_val, "test": df_test}
pd.to_pickle(df_splits, "dataset_splits.pkl")

print("✅ Data preparation complete: Train, Val, and Test splits saved as dataframe.")

os.system("git add .gitignore dvc.lock dvc.yaml") 
