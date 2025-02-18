import torch
import time
import psutil  # 用于监测 CPU 内存
import gc  # 进行垃圾回收
import os
import sys
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

# **确保可以正确导入 datasets**
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
sys.path.append(os.path.join(script_dir, ".."))  # 添加上一级目录到 Python 路径

# **导入 BraTS 数据集类**
from datasets.BraTSDataset import BraTSDataset
# **导入 UNet3D 结构**
from models.UNet3D import UNet3D

# **设备检查**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **加载模型**
model = UNet3D(in_channels=4, out_channels=1)  # BraTS 需要 4 通道输入，1 通道输出
model.load_state_dict(torch.load(
    "/Users/ranxu/Desktop/DIU-Net-main/scripts/logs/20250124_192606/best_model_state_dict.pt",
    map_location=device
))
model.to(device)
model.eval()  # 进入评估模式

# **数据集路径**
dataset_path = "/Users/ranxu/Desktop/DIU-Net-main/Dataset/BraTS/最终分类/最终"

# **加载测试数据集**
test_dataset = BraTSDataset(root_dir=dataset_path, mode="test")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# **初始化计数器**
inference_times = []  # 记录推理时间
dice_scores = []  # 记录 Dice 系数

# **计算推理时间 & 最大显存占用**
if torch.cuda.is_available():
    torch.cuda.reset_max_memory_allocated(device)  # 重置显存统计

with torch.no_grad():
    for images, masks in tqdm(test_dataloader, desc="Evaluating"):
        images, masks = images.to(device), masks.to(device)

        # **记录推理时间**
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        inference_times.append(end_time - start_time)

        # **计算 Dice 系数**
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()  # 二值化
        dice_score = (2.0 * (outputs * masks).sum()) / (outputs.sum() + masks.sum() + 1e-8)
        dice_scores.append(dice_score.item())

# **计算平均 Dice**
avg_dice = sum(dice_scores) / len(dice_scores)
print(f"BraTS Average Dice Coefficient: {avg_dice:.4f}")

# **计算平均推理时间**
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")

# **获取 GPU 最高显存占用**
if torch.cuda.is_available():
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 转换成 MB
    print(f"Maximum GPU Memory Allocated: {max_memory:.2f} MB")
