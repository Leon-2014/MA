# ---------------------------------------------------------------------
# To run this script, run `python -m scripts.train`
# To monitor Tensorboard, run `tensorboard serve --logdir runs/`
#
# Notes:
# For DATASET_DIR, the path starts from project directory
# Ensure that file names are identical for a pair of image and image mask
# ---------------------------------------------------------------------
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision import transforms
from diunet import DIUNet
from utils import ImageSegmentationDataset, EarlyStopper, Logger, BinaryMIOU
import torch.cuda
import gc
import os
from datetime import datetime

# ---------------------------------------------
# Training preparation
# ---------------------------------------------
DATASET_DIR = "/Users/ranxu/Desktop/DIU-Net-main/Dataset/CT血管分割/最终分类"

PARAMS = {
    "description": "DIU-Net trained on original data",
    "max_epochs": 120,
    "batch_size": 5,
    "learning_rate": 1e-5,
    "model_channel_scale": 0.5,
    "dense_block_depth_scale": 0.25,
}

# Check GPU availability
if torch.cuda.is_available():
    
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.8)
else:
    device = torch.device("cpu")

# model configuration
model = DIUNet(
    channel_scale=PARAMS["model_channel_scale"],
    dense_block_depth_scale=PARAMS["dense_block_depth_scale"],
)
model.to(device)
PARAMS["parameter_count"] = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {PARAMS['parameter_count']}")


# training configuration and hyperparameters
optimizer = Adam(model.parameters(), lr=PARAMS["learning_rate"], betas=(0.9, 0.999))
loss_fn = nn.BCELoss()
miou_metric = BinaryMIOU(device=device)

# ---------------------------------------------
# Dataset preparation
# ---------------------------------------------
# transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

transforms = transforms.Compose([ transforms.ToTensor(), # 转换为张量并缩放到 [0, 1] 
                                 transforms.ConvertImageDtype(torch.float32) # 确保数据类型为 torch.float32 
                                 ])
# create datasets
train_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/train/images",
    f"{DATASET_DIR}/train/masks",
    transforms,
    transforms,
    img_size=(512, 512)
)
val_dataset = ImageSegmentationDataset(
    f"{DATASET_DIR}/test/images",
    f"{DATASET_DIR}/test/masks",
    transforms,
    transforms,
    img_size=(512, 512)
)

train_dataloader = DataLoader(
    train_dataset, batch_size=PARAMS["batch_size"], shuffle=True
)
val_dataloader = DataLoader(val_dataset, batch_size=PARAMS["batch_size"])

# ---------------------------------------------
# Model training
# ---------------------------------------------
logger = Logger()
writer = SummaryWriter()
best_val_miou = float("-inf")

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"./logs/{run_name}"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(PARAMS["max_epochs"]):
    metrics = {
        "train_loss_sum": 0,
        "train_running_loss": 0,
        "train_iou_sum": 0,
        "train_running_iou": 0,
        "val_loss_sum": 0,
        "val_running_loss": 0,
        "val_iou_sum": 0,
        "val_running_iou": 0,
    }

    # start training loop
    model.train()
    for train_batch_idx, (train_imgs, train_img_masks) in enumerate(
        pbar := tqdm(train_dataloader)
    ):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        train_imgs = train_imgs.to(device)
        train_img_masks = train_img_masks.to(device)
        
        train_preds = model(train_imgs)
        loss: torch.Tensor = loss_fn(train_preds, train_img_masks)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics["train_loss_sum"] += loss.item()
        metrics["train_running_loss"] = metrics["train_loss_sum"] / (train_batch_idx + 1)
        metrics["train_iou_sum"] += miou_metric(train_preds, train_img_masks)
        metrics["train_running_iou"] = metrics["train_iou_sum"] / (train_batch_idx + 1)

        pbar.set_description(
            f"Epoch: {epoch+1}, Train Loss: {metrics['train_running_loss']}, Train mIoU: {metrics['train_running_iou']}"
        )

        del train_preds
        torch.cuda.empty_cache()

    # start evaluation loop
    model.eval()
    for val_batch_idx, (val_imgs, val_img_masks) in enumerate(
        pbar := tqdm(val_dataloader)
    ):
        val_imgs = val_imgs.to(device)
        val_img_masks = val_img_masks.to(device)

        with torch.no_grad():
            val_preds = model(val_imgs)
            loss = loss_fn(val_preds, val_img_masks)

        metrics["val_loss_sum"] += loss.item()
        metrics["val_running_loss"] = metrics["val_loss_sum"] / (val_batch_idx + 1)
        metrics["val_iou_sum"] += miou_metric(val_preds, val_img_masks)
        metrics["val_running_iou"] = metrics["val_iou_sum"] / (val_batch_idx + 1)

        pbar.set_description(
            f"Epoch: {epoch+1}, Val Loss: {metrics['val_running_loss']}, Val mIoU: {metrics['val_running_iou']}"
        )

        del val_preds
        torch.cuda.empty_cache()

    # log results
    writer.add_scalar("loss/train", metrics["train_running_loss"], epoch)
    writer.add_scalar("loss/val", metrics["val_running_loss"], epoch)
    writer.add_scalar("mIoU/train", metrics["train_running_iou"], epoch)
    writer.add_scalar("mIoU/val", metrics["val_running_iou"], epoch)

    # save best model
    if metrics["val_running_iou"] > best_val_miou:
        best_val_miou = metrics["val_running_iou"]
        PARAMS["best_epoch"] = epoch + 1
        torch.save(
            model.state_dict(), 
            os.path.join(save_dir, "best_model_state_dict.pt")
        )
        
        with open(os.path.join(save_dir, "training_params.txt"), "w") as f:
            for key, value in PARAMS.items():
                f.write(f"{key}: {value}\n")

# save final log
writer.flush()
writer.close()

PARAMS["epochs_trained"] = epoch + 1
logger.save_run(PARAMS)
