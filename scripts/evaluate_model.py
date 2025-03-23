import json
import os
import pickle
import sys
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dvclive import Live
# Define image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

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


def evaluate(model, dataloader, live):
    """
    Evaluate the model on the test dataset and log results using dvclive.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for test set.
        live (dvclive.Live): DVCLive instance for logging.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Log metrics with dvclive
    live.summary["test/accuracy"] = accuracy
    for class_label, metrics in class_report.items():
        if isinstance(metrics, dict):  # Ignore 'accuracy' key at the top level
            for metric_name, value in metrics.items():
                live.summary[f"test/{class_label}/{metric_name}"] = value

    # Save and log confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Test")
    cm_filename = "confusion_matrix_test.png"
    plt.savefig(cm_filename)
    plt.close()

    live.log_image(cm_filename, cm_filename)

    return {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON serialization
    }


def main():
    dataset_path = "dataset_splits.pkl"

    # Load the model from root directory
    model = torch.load("tuned_model.pth", weights_only=False)  # Load model from root

    # Load dataset_splits.pkl
    with open(dataset_path, "rb") as f:
        data_splits = pickle.load(f)

    df_test = data_splits["test"]
    
    # Create test dataset and dataloader
    test_dataset = ImageDataset(df_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate test set using dvclive
    with Live("eval") as live:
        results = evaluate(model, test_dataloader, live)

    # Save results to JSON
    with open("evaluation_report.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
