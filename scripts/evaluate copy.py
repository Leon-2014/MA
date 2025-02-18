import torch
import time
import psutil  # 监测 CPU 内存
import gc  # 垃圾回收
from diunet import DIUNet  # 假设你的模型类是 DIUNet
from utils import BinaryMIOU, ImageSegmentationDataset  # 自定义评估指标和数据集类
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision import transforms 

# **设备设置**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **加载模型**
model = DIUNet(channel_scale=0.5, dense_block_depth_scale=0.25)
model.load_state_dict(torch.load(
    "/Users/ranxu/Desktop/DIU-Net-main/scripts/logs/20250124_192606/best_model_state_dict.pt",
    map_location=device  # **确保在 CPU 或 GPU 上正确加载**
))
model.to(device)
model.eval()  # 设置为评估模式

# **数据集路径**
dataset_paths = {
    "CT血管分割": "/Users/ranxu/Desktop/DIU-Net-main/Dataset/CT血管分割/最终分类",
    "CT肺分割": "/Users/ranxu/Desktop/DIU-Net-main/Dataset/CT肺分割/最终分类"
}

# **数据预处理**
transforms = transforms.Compose([
    transforms.ToTensor(),  # 转换为 Tensor 并缩放到 [0, 1]
    transforms.ConvertImageDtype(torch.float32)  # 确保数据类型为 float32
])

# **初始化评估指标**
metric = BinaryMIOU(device=device)  # 计算二分类的 mIoU/Dice

# **遍历数据集**
for dataset_name, dataset_path in dataset_paths.items():
    print(f"\nEvaluating dataset: {dataset_name}")

    # **加载数据集**
    test_dataset = ImageSegmentationDataset(
        f"{dataset_path}/test/images",
        f"{dataset_path}/test/masks",
        transforms,
        transforms,
        img_size=(512, 512)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # **主脚本中调试**
    image, mask = test_dataset[0]  # 加载第一个样本
    print(f"Image type: {type(image)}, Image shape: {image.shape}")
    print(f"Mask type: {type(mask)}, Mask shape: {mask.shape}")

    # **推理和评估**
    baseline_dice_scores = []
    inference_times = []
    max_memory = 0  # 记录最大内存

    # **仅在 GPU 可用时重置显存统计**
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated(device)

    with torch.no_grad():
        for test_imgs, test_img_masks in tqdm(test_dataloader, desc=f"Evaluating {dataset_name}"):
            test_imgs = test_imgs.to(device)
            test_img_masks = test_img_masks.to(device)

            # **计算推理时间**
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                test_preds = model(test_imgs)
                end_event.record()

                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
            else:
                start_time = time.time()
                test_preds = model(test_imgs)
                end_time = time.time()
                inference_time = (end_time - start_time) / len(test_imgs)

            inference_times.append(inference_time)

            # **计算 Dice 分数**
            dice_score = metric(test_preds, test_img_masks)
            baseline_dice_scores.append(dice_score)

            # **仅在 GPU 可用时测量显存**
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                max_memory = max(max_memory, torch.cuda.max_memory_allocated(device) / (1024 ** 2))  # MB

            # **释放无用变量，避免显存泄漏**
            del test_imgs, test_img_masks, test_preds
            torch.cuda.empty_cache()  # 清理 GPU 缓存
            gc.collect()  # 触发 Python 垃圾回收

    # **输出当前数据集的平均 Dice**
    avg_dice = sum(baseline_dice_scores) / len(baseline_dice_scores)
    print(f"{dataset_name} Average Dice Coefficient: {avg_dice:.4f}")

    # **输出推理时间**
    avg_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average Inference Time per Image: {avg_inference_time:.4f} seconds")

    # **测量 GPU 或 CPU 内存占用**
    if torch.cuda.is_available():
        print(f"Maximum GPU Memory Allocated: {max_memory:.2f} MB")
    else:
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024 ** 2)  # MB
        print(f"CPU Memory Usage: {cpu_memory:.2f} MB")
