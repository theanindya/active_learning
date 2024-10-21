import logging
import os
from enum import Enum, auto
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sensation.models.segmentation import (
    DeepLabv3Plus,
    PSPNet,
    SegModel,
    UNet,
    UnetPlusPlus,
)
from sensation.train.data import Cityscapes, Mapillary, SensationDS
from sensation.utils import city_utils, map_utils

logger = logging.getLogger(__name__)

class LossFunction(Enum):
    JACCARD = auto()
    DICE = auto()
    TVERSKY = auto()
    FOCAL = auto()
    CROSS = auto()
    LOVASZ = auto()
    SOFTBCE = auto()
    SOFTCROSS = auto()
    MCC = auto()
    DICE_CROSS = auto()
    WEIGHTED_DICE_CROSS = auto()

class LossMode(Enum):
    BINARY = auto()
    MULTICLASS = auto()
    MULTILABEL = auto()

def get_loss(
    loss_func: LossFunction = LossFunction.DICE, 
    mode: LossMode = LossMode.MULTICLASS,
    num_classes: int = 13
):
    def lovasz_grad(gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if len(jaccard) > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
    
    def flatten_probas(probas, labels):
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        return probas, labels
    
    def lovasz_softmax_flat(probas, labels, classes='present'):
        if probas.numel() == 0:
            return probas * 0.
        C = probas.size(1)
        losses = []
        for c in range(C):
            fg = (labels == c).float()
            if (classes == 'present' and fg.sum() == 0):
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
        return torch.mean(torch.stack(losses))
    loss_mode = None
    loss = None

    if mode == LossMode.BINARY:
        loss_mode = smp.losses.constants.BINARY_MODE
    elif mode == LossMode.MULTICLASS:
        loss_mode = smp.losses.constants.MULTICLASS_MODE
    elif mode == LossMode.MULTILABEL:
        loss_mode = smp.losses.constants.MULTILABEL_MODE
    else:
        err_msg = f"Unsupported loss mode: {mode}"
        raise ValueError(err_msg)

    if loss_func == LossFunction.WEIGHTED_DICE_CROSS:
        class_counts = torch.tensor([264, 322, 36, 179, 153, 255, 92, 33, 140, 67, 74, 50, 36])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
        
        dice_loss = smp.losses.DiceLoss(mode=loss_mode)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        def weighted_loss(pred, target):
            # Move weights to the same device as pred
            if ce_loss.weight is not None:
                ce_loss.weight = ce_loss.weight.to(pred.device)
            return dice_loss(pred, target) + ce_loss(pred, target)
        
        loss = weighted_loss

    elif loss_func == LossFunction.JACCARD:
        loss = smp.losses.JaccardLoss(mode=loss_mode)
    elif loss_func == LossFunction.DICE:
        loss = smp.losses.DiceLoss(mode=loss_mode)
    elif loss_func == LossFunction.SOFTCROSS:
        loss = smp.losses.SoftCrossEntropyLoss(reduction='mean', smooth_factor=0.4, ignore_index=-100, dim=1)
    elif loss_func == LossFunction.TVERSKY:
        loss = smp.losses.TverskyLoss(mode=loss_mode, alpha=0.7, beta=0.3, gamma=1.0)
    elif loss_func == LossFunction.CROSS:
        class_counts = torch.tensor([264, 322, 36, 179, 153, 255, 92, 33, 140, 67, 74, 50, 36])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize weights
        loss = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_func == LossFunction.FOCAL:
        loss = smp.losses.FocalLoss(
            mode=loss_mode,
            alpha=None,
            gamma=2.0,
            ignore_index=None,
            reduction="mean",
            normalized=False,
            reduced_threshold=None,
        )
    elif loss_func == LossFunction.LOVASZ:
        def lovasz_loss(preds, labels):
            return lovasz_softmax_flat(*flatten_probas(preds, labels))
        loss = lovasz_loss
    elif loss_func == LossFunction.DICE_CROSS:
        dice_loss = smp.losses.DiceLoss(mode=loss_mode)
        ce_loss = nn.CrossEntropyLoss()
        loss = lambda pred, target: dice_loss(pred, target) + ce_loss(pred, target)
    else:
        err_msg = f"Not supported loss function: {loss_func}"
        raise ValueError(err_msg)

    return loss



def create_seg_model(
    model_arc: str = None,
    epochs: int = 1,
    num_classes: int = 8,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
    ckpt_path: str = "",
    loss=None,
    train_data: Dataset = None,
    val_data: Dataset = None,
    test_data: Dataset = None,
    weight_decay: float = 0,
):
    logger.info("Starting to create segmentation model.")
    base_model = None
    model_name = None
    encoder_name = None
    model_output = None
    parts = model_arc.split(":")
    if len(parts) == 3:
        model_name = parts[0]
        encoder_name = parts[1]
        model_output = int(parts[2])
        logger.debug(
            f"Detected segmentation model: {model_name} with encode: {encoder_name}."
        )
    else:
        error_msg = f"The model architecture is not correct defined in: {model_arc}"
        raise ValueError(error_msg)

    if model_name.startswith("deeplabv3plus"):
        base_model = DeepLabv3Plus(num_classes=model_output, encoder_name=encoder_name)
    elif model_name.startswith("pspnet"):
        base_model = PSPNet(num_classes=model_output, encoder_name=encoder_name)
    elif model_name.startswith("unetpp"):
        base_model = UnetPlusPlus(num_classes=model_output, encoder_name=encoder_name)
    elif model_name.startswith("unet"):
        base_model = UNet(num_classes=model_output, encoder_name=encoder_name)
    else:
        error_msg = f"Unsupported model architecture: {model_arc}."
        raise ValueError(error_msg)

    model = SegModel(
        num_classes=model_output,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        base_model=base_model,
        loss=loss,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        weight_decay=weight_decay,
    )

    if os.path.exists(ckpt_path):
        model = load_checkpoint(model, ckpt_path, exclude_last=False)

        if num_classes > model_output:
            model.whole_model.arc.segmentation_head = create_segmentation_head(
                model, model_name, num_classes
            )
            model.update_model(num_classes)

    return model

def get_augmentations(augmentation_list):
    aug_compose = [
        A.Resize(height=((996 - 1) // 32 + 1) * 32, width=((1328 - 1) // 32 + 1) * 32)
    ]
    if augmentation_list:
        for aug in augmentation_list:
            if aug == 'rotate':
                aug_compose.append(A.Rotate(limit=30, p=0.5))
            elif aug == 'flip':
                aug_compose.append(A.HorizontalFlip(p=0.5))
            elif aug == 'scale':
                aug_compose.append(A.RandomScale(scale_limit=0.1, p=0.5))
    
    aug_compose.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return A.Compose(aug_compose)

def prepare_cityscapes(
    root_path: str,
    batch_size: int,
    augmentations: list = None,
):
    train_transform = get_augmentations(augmentations) if augmentations else city_utils.train_transform
    
    train_dataset = Cityscapes(
        root_path,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=train_transform,
        mask_transform=city_utils.convert_input_masks,
    )

    val_dataset = Cityscapes(
        root_path,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=city_utils.val_transform,
        mask_transform=city_utils.convert_input_masks,
    )

    test_dataset = Cityscapes(
        root_path,
        split="test",
        mode="fine",
        target_type="semantic",
        transform=city_utils.val_transform,
        mask_transform=city_utils.convert_input_masks,
    )

    return train_dataset, val_dataset, test_dataset

def prepare_mapillary(root_path: str, batch_size: int, augmentations: list = None):
    train_path = os.path.join(root_path, "training")
    val_path = os.path.join(root_path, "validation")
    test_path = os.path.join(root_path, "testing")
    
    train_transform = get_augmentations(augmentations) if augmentations else map_utils.train_transform
    
    train_dataset = Mapillary(
        train_path,
        transform=train_transform,
        target_transform=map_utils.convert_input_masks,
    )
    val_dataset = Mapillary(
        val_path,
        transform=map_utils.val_transform,
        target_transform=map_utils.convert_input_masks,
    )

    test_dataset = Mapillary(
        test_path,
        transform=map_utils.val_transform,
        target_transform=map_utils.convert_input_masks,
    )

    return train_dataset, val_dataset, test_dataset

def prepare_sensation(
    root_dir: str,
    batch_size: int,
    image_height: int = 640,
    image_width: int = 800,
    augmentations: list = None,
):
    train_transform = get_augmentations(augmentations) if augmentations else None
    
    train_dataset = SensationDS(
        root_dir=root_dir,
        split="train",
        image_height=image_height,
        image_width=image_width,
        transform=train_transform,
    )

    val_dataset = SensationDS(
        root_dir=root_dir,
        split="val",
        image_height=image_height,
        image_width=image_width,
    )

    test_dataset = SensationDS(
        root_dir=root_dir,
        split="test",
        image_height=image_height,
        image_width=image_width,
    )

    return train_dataset, val_dataset, test_dataset

def load_checkpoint(model, ckpt_path, exclude_last=True):
    pass

def create_segmentation_head(model, model_name, num_classes):
    pass
