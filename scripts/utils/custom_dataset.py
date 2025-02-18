import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ImageSegmentationDataset(Dataset):
    def __init__(self, img_dir: str, img_mask_dir: str, transform=None, mask_transform=None, img_size=None):
        self.img_dir = os.path.abspath(img_dir)  # 转换为绝对路径
        self.img_mask_dir = os.path.abspath(img_mask_dir)  # 转换为绝对路径
        self.transform = transform or T.ToTensor()  # 默认将图像转换为 Tensor
        self.mask_transform = mask_transform or T.ToTensor()
        self.img_size = img_size

        


        # 检查路径是否存在
        if not os.path.exists(self.img_dir):
            raise RuntimeError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.img_mask_dir):
            raise RuntimeError(f"Mask directory not found: {self.img_mask_dir}")

        # 加载文件列表
        self.img_files = [
            f for f in os.listdir(self.img_dir) if f.lower().endswith(('.tif', '.tiff', '.png'))
        ]
        self.mask_files = [
            f for f in os.listdir(self.img_mask_dir) if f.lower().endswith(('.tif', '.tiff', '.png'))
        ]

        if len(self.img_files) == 0:
            raise RuntimeError(f"No valid image files found in {self.img_dir}")
        if len(self.mask_files) == 0:
            raise RuntimeError(f"No valid mask files found in {self.img_mask_dir}")
        if len(self.img_files) != len(self.mask_files):
            raise AssertionError(
                f"Number of images ({len(self.img_files)}) does not match number of masks ({len(self.mask_files)})"
            )
        
       

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.img_mask_dir, self.mask_files[idx])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        


        # 使用 Pillow 读取图像
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 如果指定了 img_size，调整图像大小
        if self.img_size:
            img = img.resize(self.img_size, Image.BILINEAR)
            mask = mask.resize(self.img_size, Image.NEAREST)

        # 应用 transforms，将图像和掩码转换为 Tensor
        img = self.transform(img)
        mask = self.mask_transform(mask)



        return img, mask
