import os
import re


def get_best_checkpoint(path_to_checkpoint: str) -> str:
    """Checks if checkpoints exist in given folder.
    If exist the method returns best checkpoint
    built on val loss and iou.
    """
    if not os.path.exists(path_to_checkpoint):
        return None
    best_loss = float("inf")
    best_iou = 0
    best_checkpoint = None

    # Regex to match the filenames and extract loss and IOU
    pattern = re.compile(r"epoch=\d+-val_loss=([0-9.]+)-val_iou=([0-9.]+)\.ckpt")

    for filename in os.listdir(path_to_checkpoint):
        match = pattern.match(filename)
        if match:
            loss, iou = map(float, match.groups())
            # Select the checkpoint with the lowest validation loss and then the highest IOU
            if loss < best_loss or (loss == best_loss and iou > best_iou):
                best_loss, best_iou = loss, iou
                best_checkpoint = filename

    if best_checkpoint:
        return os.path.join(path_to_checkpoint, best_checkpoint)
    else:
        return None
