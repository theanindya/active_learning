""" Define here tools to use during training."""
import pytorch_lightning as pl
import torch


def freeze_layers(model: pl.LightningModule) -> pl.LightningModule:
    """
    This module freezes the last layers of a ResNet.
    The layer4 parameter contains the last layers of a ResNet.
    Like layer4.0.conf
    Paramertrs:
    - model: pl.LightningModule the model to freeze layers.

    Returns:
    - model: pl.LightningModule model with freezed layers
    """
    # First freeze all layers
    for param in model.whole_model.arc.encoder.parameters():
        param.requires_grad = False

    for param in model.whole_model.arc.encoder.layer4.parameters():
        param.requires_grad = True

    return model


def change_model_output(model, new_num_output: int):
    """
    This method changes the last output layer
    to the desired new layer if needed
    """
    # First check current number
    current_num_output = model.segmentation_head[0].out_channels

    if current_num_output != new_num_output:
        model.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=model.segmentation_head[0].in_channels,
            out_channels=new_num_output,
            kernel_size=model.segmentation_head[0].kernel_size,
            stride=model.segmentation_head[0].stride,
            padding=model.segmentation_head[0].padding,
        )

    return model
