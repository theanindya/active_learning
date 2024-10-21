import torch
from pytorch_lightning import Trainer
from sensation.train import builder
from sensation.train.data import SensationDS
from torch.utils.data import DataLoader
from sensation.models.segmentation import SegModel  # Make sure to import SegModel

def evaluate_on_test_set(checkpoint_path, test_dataset):
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create the base model
        base_model = builder.create_seg_model(
            model_arc="unet:inceptionresnetv2:13",
            num_classes=13,
        )
        
        # Create SegModel and load state dict
        model = SegModel(
            num_classes=13,
            base_model=base_model.whole_model,
            batch_size=16,
            learning_rate=1e-4,
            epochs=100,
            loss=builder.get_loss(builder.LossFunction.DICE, num_classes=13),
            test_data=test_dataset
        )
        model.load_state_dict(checkpoint['state_dict'])
        
        # Create a trainer
        trainer = Trainer(accelerator='gpu', devices=1, logger=False)
        
        # Create test dataloader
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Run the test
        test_results = trainer.test(model, dataloaders=test_loader)
        
        return test_results
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_path}: {str(e)}")
        return None



# Load your test dataset
test_dataset = SensationDS(
    root_dir="/home/woody/iwi5/iwi5088h/sensation",
    split="test",
    image_height=640,
    image_width=800
)

# List of your top 3 checkpoint paths
top_checkpoints = [
    "/home/woody/iwi5/iwi5088h/segmentation/checkpoints/n9epoch=00-val_iouval_iou=0.4272.ckpt",
    "/home/woody/iwi5/iwi5088h/segmentation/checkpoints/n9epoch=00-val_iouval_iou=0.4219.ckpt",
    "/home/woody/iwi5/iwi5088h/segmentation/checkpoints/n9epoch=00-val_iouval_iou=0.4213.ckpt"
]

# Evaluate each checkpoint
for ckpt in top_checkpoints:
    results = evaluate_on_test_set(ckpt, test_dataset)
    if results:
        print(f"Results for {ckpt}:")
        print(results)
    else:
        print(f"No results obtained for {ckpt}")