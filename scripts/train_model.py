import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from model import CNN  # Define CNN architecture in model.py
import os
import io


CONFIG_PATH = "params.yaml"

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

if os.path.exists("tuned_model.pth"):
    os.remove("tuned_model.pth")
    print("Deleted existing tuned_model.pth")

lr_values = config["model"]["lr"]
conv_layers_values = config["model"]["conv_layers"]
seed = config["seed"]

# Load dataset splits
df_splits = pd.read_pickle("dataset_splits.pkl")
df_train = df_splits["train"]
df_val = df_splits["val"]

# Image transformation
transform = transforms.Compose([transforms.ToTensor()])

# Dataset class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.labels = sorted(df["label"].unique())
        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_bytes = self.df.iloc[idx]["image_bytes"]
        label = self.label_map[self.df.iloc[idx]["label"]]
        
        # Convert byte data to PIL Image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = self.transform(image)
        
        return image, label

# Create datasets
train_dataset = ImageDataset(df_train)
val_dataset = ImageDataset(df_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train with hyperparameter tuning
best_model = None
best_acc = 0.0

for lr in lr_values:
    for conv_layers in conv_layers_values:
        print(f"Training model with LR={lr}, Conv Layers={conv_layers}")

        model = CNN(conv_layers=conv_layers)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Validation loop
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model = model


# Save the best model
if best_model:
    torch.save(best_model, "tuned_model.pth") 

print("✅ Model training & tuning completed.")

os.system("git add .gitignore dvc.lock dvc.yaml") 
