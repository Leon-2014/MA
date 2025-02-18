import torch
from diunet import DIUNet  # 假设你的模型类是 DIUNet
from utils import BinaryMIOU, ImageSegmentationDataset  # 自定义评估指标和数据集类
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision import transforms 
# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = DIUNet(channel_scale=0.5, dense_block_depth_scale=0.25)
model.load_state_dict(torch.load("/Users/ranxu/Desktop/DIU-Net-main/scripts/logs/20250124_192606/best_model_state_dict.pt"))  # 替换为你的模型权重路径
model.to(device)
model.eval()  # 设置为评估模式

# 数据集路径
dataset_paths = {
    "CT血管分割": "/Users/ranxu/Desktop/DIU-Net-main/Dataset/CT血管分割/最终分类",
    "CT肺分割": "/Users/ranxu/Desktop/DIU-Net-main/Dataset/CT肺分割/最终分类"
}

# 数据预处理
# transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])  # 数据预处理
transforms = transforms.Compose([ transforms.ToTensor(), # 转换为张量并缩放到 [0, 1] 
                                 transforms.ConvertImageDtype(torch.float32) # 确保数据类型为 torch.float32 
                                 ])
# 初始化评估指标
metric = BinaryMIOU(device=device)  # 计算二分类的 mIoU/Dice

# 遍历数据集
for dataset_name, dataset_path in dataset_paths.items():
    print(f"\nEvaluating dataset: {dataset_name}")

    # 加载数据集
    test_dataset = ImageSegmentationDataset(
        f"{dataset_path}/test/images",
        f"{dataset_path}/test/masks",
        transforms,
        transforms,
        img_size=(512, 512)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # 主脚本中调试
    image, mask = test_dataset[0]  # 加载第一个样本
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")


    # 推理和评估
    baseline_dice_scores = []  # 存储每批的 Dice 分数
    with torch.no_grad():  # 禁用梯度计算，提升推理效率
        for test_imgs, test_img_masks in tqdm(test_dataloader, desc=f"Evaluating {dataset_name}"):
            test_imgs = test_imgs.to(device)  # 将图像数据移动到 GPU/CPU
            test_img_masks = test_img_masks.to(device)  # 将标签数据移动到 GPU/CPU

            test_preds = model(test_imgs)  # 模型推理，得到预测结果
            dice_score = metric(test_preds, test_img_masks)  # 计算 Dice 分数
            baseline_dice_scores.append(dice_score)  # 保存当前批次的分数

    # 输出当前数据集的平均 Dice
    avg_dice = sum(baseline_dice_scores) / len(baseline_dice_scores)  # 计算平均 Dice
    print(f"{dataset_name} Average Dice Coefficient: {avg_dice:.4f}")
