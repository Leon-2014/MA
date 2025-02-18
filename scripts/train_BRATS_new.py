import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 将 models 文件夹路径添加到 sys.path
print(parent_dir)
sys.path.append(parent_dir)
from models.UNet3D import UNet3D
import torch.nn.functional as F
import gc
import sys
from tqdm import tqdm


# 获取 scripts 目录的上一级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 把 datasets 所在的路径加入 Python 搜索路径
sys.path.append(BASE_DIR)

from datasets.BraTSDataset import BraTSDataset
# 设备检查
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据集路径
DATASET_DIR = "/Users/ranxu/Desktop/DIU-Net-main/Dataset/BraTS/最终分类/最终"

# 训练主函数
def train():
    # **加载数据**
    train_dataset = BraTSDataset(root_dir=DATASET_DIR, mode="train")
    val_dataset = BraTSDataset(root_dir=DATASET_DIR, mode="val")

    # **创建 DataLoader**
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=2, num_workers=2)

    # **初始化 3D U-Net**
    model = UNet3D(in_channels=4, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")

    # **训练循环**
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # **检查尺寸匹配**
            if outputs.shape != targets.shape:
                print(f"Size mismatch: Output {outputs.shape}, Target {targets.shape}")
                targets = F.interpolate(targets, size=outputs.shape[2:], mode="nearest")  # 调整大小

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            del inputs, targets, outputs
            torch.cuda.empty_cache()
            gc.collect()

        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # **保存最佳模型**
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train()
