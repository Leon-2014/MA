import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
# Dataset definition for BraTS
class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, mask_transforms=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*/*.png")))
        print('image_paths',len(self.image_paths))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*/*.png")))
        print('mask_paths',len(self.mask_paths))
        self.transforms = transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        # Load image (multi-channel)
        image_path = self.image_paths[idx]
        image_data = np.array(Image.open(image_path)) # (H, W, C)

        # Load mask
        mask_path = self.mask_paths[idx]
        mask_data = np.array(Image.open(mask_path))  # (H, W)

        # Convert to NumPy
        image = np.array(image_data, dtype=np.float32)
        mask = np.array(mask_data, dtype=np.uint8)

        # Normalize image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        if self.mask_transforms:
            mask = self.mask_transforms(mask)

        # Convert to tensors
        image = torch.tensor(image).permute(2, 0, 1)  # (C, H, W)
        mask = torch.tensor(mask).long()  # Multi-class segmentation

        return image, mask

# Metrics
def dice_coefficient(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def jaccard_index(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Model definition (example: UNet)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training loop
def train_model():
    DATASET_DIR = "/Users/ranxu/Desktop/DIU-Net-main/Dataset/BraTS/最终分类/"

    # Datasets and loaders
    train_dataset = BraTSDataset(
        f"{DATASET_DIR}/train/images",
        f"{DATASET_DIR}/train/masks",
    )
    val_dataset = BraTSDataset(
        f"{DATASET_DIR}/test/images",
        f"{DATASET_DIR}/test/masks",
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    if torch.cuda.is_available():
    

        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    # Model, loss, optimizer
    model = SimpleUNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(1, 101):
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch}, Training Loss: {train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_jaccard = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.cuda(), masks.cuda()

                outputs = model(images)
                loss = loss_fn(outputs, masks)

                # Metrics
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item()
                val_dice += dice_coefficient(preds, masks).item()
                val_jaccard += jaccard_index(preds, masks).item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, \
                Dice: {val_dice / len(val_loader):.4f}, \
                Jaccard: {val_jaccard / len(val_loader):.4f}")

if __name__ == "__main__":
    train_model()