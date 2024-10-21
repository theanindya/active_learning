import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes as TorchCityscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class Mapillary(Dataset):
    def __init__(self, data_path, transform, target_transform, num_classes=13):
        self.data_path = data_path
        self.img_names = sorted(os.listdir(os.path.join(data_path, "images")))
        self.mask_names = sorted(os.listdir(os.path.join(data_path, "masks")))
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes

    def __getitem__(self, idx):
        # load image
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_path, "images", img_name)
        image = Image.open(img_path)
        image = np.array(image)

        # load mask
        mask_name = self.mask_names[idx]
        mask_path = os.path.join(self.data_path, "masks", mask_name)
        mask = Image.open(mask_path)  # read grayscale
        mask = np.array(mask)

        # Handle invalid mask values
        mask[mask >= self.num_classes] = 0  # Set invalid values to background class

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

    def __len__(self):
        return len(self.img_names)

class Cityscapes(TorchCityscapes):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        target_type="semantic",
        mode: str = "fine",
        transform=None,
        mask_transform=None,
        num_classes=13,
    ):
        super(Cityscapes, self).__init__(
            root=dataset_path, split=split, mode=mode, target_type=target_type
        )
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

    def __getitem__(self, index):
        image, mask = super(Cityscapes, self).__getitem__(index)
        # Convert PIL image and mask into numpy arrays
        image = np.array(image)
        mask = np.array(mask, dtype=np.float32)

        # Handle invalid mask values
        mask[mask >= self.num_classes] = 0  # Set invalid values to background class

        # Apply transformations
        if self.mask_transform:
            mask = self.mask_transform(mask)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

class SensationDS(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_height: int = 640,
        image_width: int = 800,
        split: str = "train",
        num_classes: int = 13,
        transform=None,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.split = split
        self.num_classes = num_classes
        self.transform = transform

        if split == "train":
            self.root_dir = os.path.join(root_dir, "training")
        elif split == "val":
            self.root_dir = os.path.join(root_dir, "validation")
        elif split == "test":
            self.root_dir = os.path.join(root_dir, "testing")
        else:
            err_msg = f"Unsupported split: {split}. Please choose: train, val or test."
            raise ValueError(err_msg)

        images_dir = os.path.join(self.root_dir, "images")
        masks_dir = os.path.join(self.root_dir, "masks")

        self.images = sorted(
            [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
        )
        self.masks = sorted(
            [os.path.join(masks_dir, mask) for mask in os.listdir(masks_dir)]
        )

        # Define default transformations
        self.default_train_transforms = A.Compose(
            [
                A.Resize(image_height, image_width),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.default_val_transform = A.Compose(
            [
                A.Resize(image_height, image_width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure the image dimensions are divisible by 32
        h, w = image.shape[:2]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Handle invalid mask values
        mask[mask >= self.num_classes] = 0  # Set invalid values to 0 (unlabelled)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
        else:
            if self.split == "train":
                transformed = self.default_train_transforms(image=image, mask=mask)
            else:
                transformed = self.default_val_transform(image=image, mask=mask)
        
        image = transformed["image"]
        mask = transformed["mask"]

        return image, mask.long()