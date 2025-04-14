"""
File: spd/experiments/waterbird/train_resnet.py
Trains the WaterbirdResNet18 model on the Waterbirds dataset using WILDS.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from wilds import get_dataset
from models import WaterbirdResNet18
import numpy as np
import os

class WaterbirdsSubset(Dataset):
    def __init__(self, waterbird_dataset, indices, transform=None):
        self.dataset = waterbird_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        x, y, meta = self.dataset[original_idx]
        if self.transform:
            x = self.transform(x)
        return x, y, meta[0]

def main_resnet_train(
    batch_size=32,
    lr=1e-3,
    hidden_dim=512,
    num_epochs=5,
    train_size=2000,
    val_size=1000,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints"
):
    # 1) Load Waterbirds dataset (WILDS)
    waterbird_dataset = get_dataset(dataset="waterbirds", download=False)
    dataset_size = len(waterbird_dataset)
    print(f"Total dataset size: {dataset_size}")
    
    # Create random indices for train and validation without overlap
    # First shuffle all indices
    all_indices = np.arange(dataset_size)
    np.random.shuffle(all_indices)
    
    # Take the first train_size indices for training
    train_indices = all_indices[:train_size].tolist()
    
    # Take the next val_size indices for validation (ensuring no overlap)
    val_indices = all_indices[train_size:train_size+val_size].tolist()
    
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Checking for overlap: {len(set(train_indices).intersection(set(val_indices)))}")

    matched_indices = []
    for idx in range(dataset_size):
        _, label, meta = waterbird_dataset[idx]
        if label == meta[0]:
            matched_indices.append(idx)

    np.random.shuffle(matched_indices)
    train_indices = matched_indices[:train_size]
    val_indices = matched_indices[train_size:train_size + val_size]

    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Overlap between train and val: {len(set(train_indices).intersection(set(val_indices)))}")
    
    # Create transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),  # Adding data augmentation to reduce overfitting
        T.ToTensor()
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Create the training subset
    train_subset = WaterbirdsSubset(
        waterbird_dataset, 
        indices=train_indices,
        transform=train_transform
    )
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # Create a validation subset
    val_subset = WaterbirdsSubset(
        waterbird_dataset,
        indices=val_indices,
        transform=val_transform
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # 2) Initialize model
    model = WaterbirdResNet18(num_classes=2, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add learning rate scheduler to help with training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Track best validation accuracy
    best_val_acc = 0.0
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 3) Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (imgs, labels, meta) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update running loss
            running_loss += loss.item()
            
            if i % 10 == 0:
                # Calculate current batch accuracy
                batch_acc = 100 * correct / total if total > 0 else 0
                print(f"Epoch {epoch}, step {i}, loss={running_loss/(i+1):.4f}, acc={batch_acc:.2f}%")
        
        # Calculate epoch training accuracy
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels, meta in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation accuracy
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate scheduler
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch} summary:")
        print(f"  Training:   loss={running_loss/len(train_loader):.4f}, accuracy={train_acc:.2f}%")
        print(f"  Validation: loss={avg_val_loss:.4f}, accuracy={val_acc:.2f}%")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_indices': train_indices,
                'val_indices': val_indices,
            }, os.path.join(save_dir, "waterbird_resnet18_best.pth"))
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # 4) Save final checkpoint
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'train_indices': train_indices,
        'val_indices': val_indices,
    }, os.path.join(save_dir, "waterbird_resnet18_final.pth"))
    
    print(f"Training completed.")
    print(f"Final training accuracy: {train_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, best_val_acc

def load_model_from_checkpoint(checkpoint_path, model_class=WaterbirdResNet18, hidden_dim=512):
    """
    Load a model from a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_class: Model class to instantiate
        hidden_dim: Hidden dimension for the model
    
    Returns:
        The loaded model in eval mode
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = model_class(num_classes=2, hidden_dim=hidden_dim).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    
    return model

def predict(model, image_path, device=None):
    """
    Run inference on a single image
    
    Args:
        model: The PyTorch model
        image_path: Path to the image file
        device: Device to run on (if None, uses CUDA if available)
    
    Returns:
        Dictionary with prediction results
    """
    from PIL import Image
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Transform for inference
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    pred_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][pred_class].item()
    
    # For Waterbirds: class 0 = landbird, class 1 = waterbird
    pred_label = "waterbird" if pred_class == 1 else "landbird"
    
    return {
        "class_id": pred_class,
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy()
    }

if __name__ == "__main__":
    model, best_acc = main_resnet_train()