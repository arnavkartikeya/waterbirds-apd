import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from wilds import get_dataset
import numpy as np
import os
from models import ColorMNISTConvNetGAP


from glob import glob
from PIL import Image

class SpuriousMNIST(Dataset):
    """
    A lightweight wrapper around the colourised-MNIST training split that exposes
    the *spurious* background colour as an additional label.

    For every sample the dataset returns:
        X : Tensor (image)                    – shape [3, 28, 28], range [0, 1]
        Y : int    (digit class)              – 0 … 9
        Z : int    (background colour class)  – 0 = red, 1 = green

    Spurious correlation:
        • Digits 0-4 are 80 % red,    20 % green
        • Digits 5-9 are 80 % green,  20 % red
    The images on disk are expected to live in
        <root_dir>/<digit>/[red|green]_<something>.png
    """
    def __init__(
        self,
        root_dir: str,
        transform: T.Compose | None = None,
        colour_prior: float = 0.8,
        rng_seed: int | None = 0,
    ):
        self.root_dir = root_dir
        self.transform = transform or T.ToTensor()
        self.colour_prior = colour_prior

        # We will store tuples (filepath, digit_label, colour_label)
        self.data: list[tuple[str, int, int]] = []

        rng = np.random.default_rng(rng_seed)

        for digit in range(10):
            digit_dir = os.path.join(self.root_dir, str(digit))
            if not os.path.isdir(digit_dir):
                raise FileNotFoundError(f"Directory '{digit_dir}' not found.")

            # Collect *.png files for this digit
            all_pngs = glob(os.path.join(digit_dir, "*.png"))

            for fp in all_pngs:
                # File format assumed to be "<colour>_<id>.png"
                colour_str = os.path.basename(fp).split("_")[0].lower()
                if colour_str not in ("red", "green"):
                    # Skip anything that is not explicitly red / green
                    continue
                colour_label = 0 if colour_str == "red" else 1  # 0 = red, 1 = green

                # Decide whether to keep the sample based on the desired prior
                if digit <= 4:
                    # Majority red for digits 0-4
                    keep_prob = self.colour_prior if colour_label == 0 else 1 - self.colour_prior
                else:
                    # Majority green for digits 5-9
                    keep_prob = self.colour_prior if colour_label == 1 else 1 - self.colour_prior

                if rng.random() < keep_prob:
                    self.data.append((fp, digit, colour_label))

        if len(self.data) == 0:
            raise RuntimeError("No data found – check that the directory structure is correct.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        fp, digit_label, colour_label = self.data[idx]

        # Load image and convert to RGB
        img = Image.open(fp).convert("RGB")

        # Apply transform (always ending in ToTensor so we return a Tensor)
        img = self.transform(img)

        return img, digit_label, colour_label
    

def main_mnist_train(
    batch_size=32,
    lr=1e-3,
    num_epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="checkpoints"
):
    # 1) Load SpuriousMNIST dataset
    train_dataset = SpuriousMNIST(root_dir="colorized-MNIST/training", colour_prior=1.0)
    val_dataset = SpuriousMNIST(root_dir="colorized-MNIST/testing", colour_prior=1.0)

    # 2) Create DataLoaders
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3) Create model
    model = ColorMNISTConvNetGAP(num_classes=10, hidden_dim=128)
    model.to(device)
    
    # 4) Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0 
    # 5) Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels, background_colors) in enumerate(loader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            background_colors = background_colors.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if i % 10 == 0:
                # Calculate current batch accuracy
                batch_acc = 100 * correct / total if total > 0 else 0
                print(f"Epoch {epoch}, step {i}, loss={running_loss/(i+1):.4f}, acc={batch_acc:.2f}%")

        # 6) Validation loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (inputs, labels, background_colors) in enumerate(val_loader, 0): 
                inputs = inputs.to(device)
                labels = labels.to(device)
                background_colors = background_colors.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)   
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total
            print(f"Epoch {epoch} validation accuracy: {val_acc:.2f}%") 

        # 7) Save model checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved to {checkpoint_path}")   

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"Best model saved to {os.path.join(save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main_mnist_train(num_epochs=15)

        
            
            
