import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix

# Add path so backend app modules can be loaded if run directly
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.model import EmotionCNN, EMOTIONS

class FERDataset(Dataset):
    """
    Custom Dataset to load FER-2013 formatted directory of grayscale facial crops.
    """
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(EMOTIONS):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.png"):
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
                
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}")
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image as PIL Grayscale
        image = Image.open(img_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model(epochs: int = 15, batch_size: int = 32, lr: float = 0.001, progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    # Paths
    backend_dir = Path(__file__).parent.parent
    train_dir = backend_dir / "data" / "train"
    val_dir = backend_dir / "data" / "val"
    save_path = backend_dir / "app" / "model.pth"
    metrics_path = backend_dir / "app" / "metrics.json"
    
    # Ensure data directory exists. If not, raise exception (user should run generate_data first)
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Data directories do not exist. Please run generate_data.py first.")
        
    # Transforms (normalization matches PyTorch standard and augments training data)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Datasets and Loaders
    train_dataset = FERDataset(train_dir, transform=train_transform)
    val_dataset = FERDataset(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epochs": []
    }
    
    print("Beginning model training...")
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = 100.0 * correct / total
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = 100.0 * val_correct / val_total
        
        # Adjust Learning Rate
        scheduler.step(epoch_val_loss)
        
        # Store logs
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["epochs"].append(epoch)
        
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        if progress_callback:
            progress_callback({
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc
            })
        
        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint to {save_path}")
            
    # Load best weights to compute final evaluation and confusion matrix
    if save_path.exists():
        model.load_state_dict(torch.load(save_path))
        
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    # Calculate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Store history, mapping and confusion matrix into metrics.json
    metrics = {
        "history": history,
        "emotions": EMOTIONS,
        "confusion_matrix": cm.tolist(),
        "final_val_acc": float(np.mean(all_preds == np.array(all_targets)) * 100)
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model evaluation complete. Confusion matrix and logs saved to {metrics_path}")
    print(f"Training finished successfully! Model accuracy: {metrics['final_val_acc']:.2f}%")

if __name__ == "__main__":
    train_model(epochs=15, batch_size=32, lr=0.001)
