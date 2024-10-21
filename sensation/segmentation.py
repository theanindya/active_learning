import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes: int = 8, encoder_name: str = None):
        super(DeepLabv3Plus, self).__init__()
        self.arc = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, images):
        h, w = images.shape[2:]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        if h != new_h or w != new_w:
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        logits = self.arc(images)
        return logits

class UNet(nn.Module):
    def __init__(self, num_classes: int = 8, encoder_name: str = None):
        super(UNet, self).__init__()
        self.classes = num_classes
        self.arc = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            decoder_use_batchnorm=True,
            activation=None,
        )

    def forward(self, images):
        h, w = images.shape[2:]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        if h != new_h or w != new_w:
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        logits = self.arc(images)
        return logits

class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes: int = 8, encoder_name: str = None):
        super(UnetPlusPlus, self).__init__()
        self.classes = num_classes
        self.arc = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            decoder_use_batchnorm=True,
            activation=None,
        )

    def forward(self, images):
        h, w = images.shape[2:]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        if h != new_h or w != new_w:
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        logits = self.arc(images)
        return logits

class PSPNet(nn.Module):
    def __init__(self, num_classes: int = 8, encoder_name: str = None):
        super(PSPNet, self).__init__()
        self.classes = num_classes
        self.arc = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, images):
        h, w = images.shape[2:]
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        if h != new_h or w != new_w:
            images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        logits = self.arc(images)
        return logits

class SegModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 8,
        change_output: int = 0,
        base_model=None,
        batch_size: int = 8,
        learning_rate: float = 0.0001,
        epochs: int = 100,
        loss=None,
        train_data: Dataset = None,
        val_data: Dataset = None,
        test_data: Dataset = None,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.whole_model = base_model
        self.criterion = loss
        self.metrics = IoU(num_classes=num_classes)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data
        self.weight_decay = weight_decay
        print(f"SegModel initialized on: {self.device}")

    def forward(self, x):
        return self.whole_model(x)

    def training_step(self, batch, batch_idx):
        images, semantic_masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, semantic_masks.long())
        iou = self.metrics(outputs, semantic_masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def validation_step(self, batch, batch_idx):
        images, semantic_masks = batch
        outputs = self(images)
        val_loss = self.criterion(outputs, semantic_masks.long())
        val_iou = self.metrics(outputs, semantic_masks)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_iou', val_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return val_loss

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_step(self, batch, batch_idx):
        images, semantic_masks = batch
        outputs = self(images)
        test_loss = self.criterion(outputs, semantic_masks.long())
        test_iou = self.metrics(outputs, semantic_masks)
        self.log_dict({"test_loss": test_loss, "test_iou": test_iou})
        return {"test_loss": test_loss, "test_iou": test_iou}

    def test_dataloader(self, test_dataset=None):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_epoch_end(self, outputs):
        iou_per_class = self.metrics.compute()
        self.log_dict(
            {"iou_class_" + str(i): iou.item() for i, iou in enumerate(iou_per_class)},
            prog_bar=True,
        )
        return {"iou_per_class": iou_per_class}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                steps_per_epoch=len(self.train_dataloader()),
                epochs=self.epochs,
            ),
            "interval": "step",
            "frequency": 1,
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def update_model(self, new_num_classes):
        self.metrics = IoU(num_classes=new_num_classes)
        
    def on_move_to_device(self):
        print(f"SegModel moved to: {self.device}")

if __name__ == "__main__":
    model = SegModel()