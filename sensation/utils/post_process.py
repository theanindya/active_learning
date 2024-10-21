import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
def apply_morphological_operations(outputs, min_component_size=10):

    # If the outputs are logits/probabilities, take the argmax to get the class labels
    # Assuming outputs are logits with shape [batch_size, num_classes, height, width]
    if outputs.dim() == 4:
        masks = torch.argmax(outputs, dim=1).cpu().numpy()
    else:
        masks = outputs.cpu().numpy()

    # Initialize the list to hold processed masks
    processed_masks = []

    for mask in masks:
        # Ensure the mask is 2D
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(axis=0)
        elif mask.ndim == 3:
            mask = mask[0]
        
        # Optional: Morphological operations to clean up noise
        # Create a separate mask for each class to avoid mixing classes
        new_mask = np.zeros_like(mask)
        for class_id in range(mask.max() + 1):
            class_mask = (mask == class_id).astype(np.uint8) * 255
            
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
            
            # Find connected components and filter small components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                    new_mask[labels == i] = class_id
        
        processed_masks.append(new_mask)

    # Convert the list to a numpy array
    processed_masks = np.array(processed_masks)
    
    # Convert back to tensor
    processed_masks = torch.tensor(processed_masks, dtype=torch.int64)

    return processed_masks
