# This version of training_pipeline.py is specific to the SENSATION dataset. To use cityscape dataset, run the 2nd phase of the code only
import urllib.request
urllib.request.urlopen = lambda *args, **kwargs: None
import os
os.environ['ALBUMENTATIONS_DISABLE_VCHECK'] = '1'
os.environ['NO_ALBU_VCHECK'] = '1'
import albumentations
from pytorch_lightning.callbacks import EarlyStopping
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sensation.train import builder
from sensation.utils import data
from sensation.active_learning import ActiveLearningManager
from sensation.train.builder import LossFunction
import torch
torch.set_float32_matmul_precision('medium')
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training pipeline for SENSATION segmentation models with active learning.")
    parser.add_argument("--ckpt", default="/home/woody/iwi5/iwi5088h/segmentation/checkpoints",
                        help="Path where to store or load checkpoints.")
    parser.add_argument("--data_root", required=True, help="Path to the dataset to use (Cityscapes, Mapillary, or SENSATION).")
    parser.add_argument("--data_type", type=str, choices=["cityscapes", "mapillary", "sensation"], required=True,
                        help="Specify the dataset to be used for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--classes", type=int, default=13, help="Number of classes to use for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to use for each active learning iteration.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of batches to use during training.")
    parser.add_argument("--model_arc", type=str, default="unet:timm-tf_efficientnet_lite0:8",
                        help="The model architecture to use during training.")
    parser.add_argument("--precision", type=str, choices=["16", "32", "64", "bf16", "mixed"], default="32",
                        help="Precision for training.")
    parser.add_argument("--active_learning", action="store_true", help="Enable active learning")
    parser.add_argument("--al_iterations", type=int, default=5, help="Number of active learning iterations")
    parser.add_argument("--samples_per_iteration", type=int, default=100, help="Number of samples to label in each active learning iteration")
    parser.add_argument("--loss", type=lambda x: LossFunction[x.upper()], default=LossFunction.DICE, 
                        choices=list(LossFunction), help="The loss function to use.")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of epochs with no improvement after which training will be stopped")
    parser.add_argument("--augmentation", type=str, default=None,
                        help="Comma-separated list of augmentations to apply (e.g., 'rotate,flip,scale')")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs for training")
    parser.add_argument("--min_epochs", type=int, default=10, help="Minimum number of epochs for training")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--use_ensemble", action="store_true", help="Use ensemble for uncertainty estimation")
    parser.add_argument("--initial_subset_size", type=int, default=100, help="Initial size of the labeled subset")
    
    
    
    return parser.parse_args()
    

def get_augmentations(aug_string):
    if aug_string is None:
        return None
    
    aug_list = aug_string.split(',')
    augmentations = []
    
    for aug in aug_list:
        if aug == 'rotate':
            augmentations.append(A.Rotate(limit=30, p=0.5))
        elif aug == 'flip':
            augmentations.append(A.HorizontalFlip(p=0.5))
        elif aug == 'scale':
            augmentations.append(A.RandomScale(scale_limit=0.1, p=0.5))
    
    if augmentations:
        return A.Compose(augmentations + [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return None

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Main script device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    logger.info("Starting SENSATION training pipeline with active learning.")
    logger_tb = TensorBoardLogger("tb_logs", name="sensation_al")
    
    early_stop_callback = None
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val_iou',
            min_delta=0.00,
            patience=args.early_stopping_patience,
            verbose=True,
            mode='max'
        )
        args.early_stop_callback = early_stop_callback
    augmentations = get_augmentations(args.augmentation)

    # Load datasets
    if args.data_type == "cityscapes":
        train_dataset, val_dataset, _ = builder.prepare_cityscapes(args.data_root, batch_size=args.batch_size)
    elif args.data_type == "mapillary":
        train_dataset, val_dataset, _ = builder.prepare_mapillary(args.data_root, batch_size=args.batch_size)
    elif args.data_type == "sensation":
        train_dataset, val_dataset, _ = builder.prepare_sensation(root_dir=args.data_root, batch_size=args.batch_size)
    else:
        raise ValueError("Unknown dataset")

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt,
        filename='{n16epoch:02d}-val_iou{val_iou:.4f}',
        save_top_k=3,
        monitor='val_iou',
        mode='max',
        save_last=True,
        every_n_epochs=1,
        verbose=True
    )

    # Add checkpoint_callback to args
    args.checkpoint_callback = checkpoint_callback

    # Path to your best checkpoint
    checkpoint_dir = args.ckpt
    
    
    if args.active_learning:
        logger.info("Starting active learning process")
        # Initialize ActiveLearningManager with the checkpoint directory
        al_manager = ActiveLearningManager(checkpoint_dir, train_dataset, val_dataset, args)

    
        

        # Run active learning
        final_model = al_manager.run_active_learning(
    initial_subset_size=args.initial_subset_size,
    max_iterations=args.al_iterations,
    samples_per_iteration=args.samples_per_iteration,
    epochs_per_iteration=args.epochs
)
    else:
        logger.info("Starting standard fine-tuning process")
        # Load the model from the best checkpoint
        model = builder.create_seg_model(
            model_arc=args.model_arc,
            epochs=args.epochs,
            num_classes=args.classes,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            loss=builder.get_loss(args.loss),
            train_data=train_dataset,
            val_data=val_dataset,
            test_data=None,
        )
        
        # Load the best checkpoint if it exists
        best_checkpoint = os.path.join(checkpoint_dir, "best_model.ckpt")
        if os.path.exists(best_checkpoint):
            checkpoint = torch.load(best_checkpoint)
            model.load_state_dict(checkpoint['state_dict'])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Create trainer and start training
        callbacks = [checkpoint_callback]
        if early_stop_callback:
            callbacks.append(early_stop_callback)

        trainer = Trainer(
            logger=logger_tb,
            max_epochs=args.epochs,
            precision=args.precision,
            callbacks=callbacks,
            accelerator='gpu',
            devices=1,
        )
        trainer.fit(model, train_loader, val_loader)
        final_model = model

    # Save the final model
    torch.save(final_model.state_dict(), os.path.join(args.ckpt, "final_model_al.pth"))
    logger.info(f"Final model saved to {os.path.join(args.ckpt, 'final_model_al.pth')}")

if __name__ == "__main__":
    main()

# This version of training_pipeline.py is specific to the Cityscapes dataset.
# Ensure that data loaders, loss functions, and transformations are correctly set up for Cityscapes.
import os
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from sensation.train import builder, tools
from sensation.utils import data
from torch.utils.data import DataLoader

os.environ["NO_ALBU_VCHECK"] = "1"
# Define logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def loss_function_arg(string):
    try:
        return builder.LossFunction[string.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{string} is not a valid loss function.")

def main():
    parser = argparse.ArgumentParser(
        description="Training pipeline for SENSATION segmentation models."
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints",
        help="Path where to store or load checkpoints. (Default = checkpoints)",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to the dataset to use (Cityscapes, Mapillary, or SENSATION).",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["cityscapes", "mapillary", "sensation"],
        required=True,
        help="Specify the dataset to be used for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default = 0.0001)",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=11,
        help="Number of classes to use for training (default = 11)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to use for training (default = 1).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,  # Increased batch size
        help="Number of batches to use during training. (default = 16).",
    )
    parser.add_argument(
        "--model_arc",
        type=str,
        default="UNet:resnet18:11",
        help="The model architecture to use during training. (default = UNet:resnet18:11).",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Enable freezing of layers in a model.",
        default=False,
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Progress training on last epoch state if training was interrupted.",
        default=False,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test on defined checkpoint with defined testset by data type.",
        default=False,
    )
    parser.add_argument(
        "--loss",
        type=loss_function_arg,
        choices=list(builder.LossFunction),
        default=builder.LossFunction.DICE,
        help="The loss function to use.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["16", "32", "64", "bf16", "mixed"],
        default="32",
        help="Precision for training. (default = 32).",
    )

    args = parser.parse_args()

    logger.info("Starting SENSATION training pipeline.")

    logger_tb = TensorBoardLogger("tb_logs", name="mapillary_deeplab")

    if args.data_type == "cityscapes":
        train_dataset, val_dataset, _ = builder.prepare_cityscapes(
            args.data_root, batch_size=args.batch_size
        )
    elif args.data_type == "mapillary":
        train_dataset, val_dataset, _ = builder.prepare_mapillary(
            args.data_root, batch_size=args.batch_size
        )
    elif args.data_type == "sensation":
        train_dataset, val_dataset, _ = builder.prepare_sensation(
            root_dir=args.data_root, batch_size=args.batch_size
        )
    else:
        raise ValueError("Unknown dataset")

    ckpt_path = data.get_best_checkpoint(args.ckpt)
    resume_ckpt = ""
    if args.progress and ckpt_path is not None:
        logger.info(f"Starting training on last status with checkpoint: {ckpt_path}")
        resume_ckpt = ckpt_path
        ckpt_path = ""

    model = builder.create_seg_model(
        model_arc=args.model_arc,
        epochs=args.epochs,
        num_classes=args.classes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loss=builder.get_loss(args.loss),
        ckpt_path=ckpt_path,
        train_dataloader=train_dataset,
        val_dataloader=val_dataset,
        test_dataloader=None,
    )

    if args.test:
        logger.info("Starting test mode")
        trainer = Trainer()
        trainer.test(model)
        exit()

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt,
        filename="{epoch}-{val_loss:.5f}-{val_iou:.5f}",
        save_top_k=3,
        monitor="val_iou",
        mode="max",
    )

    # Freeze layers if needed
    if args.freeze:
        model = tools.freeze_layers(model)

    # Handle precision argument
    if args.precision == "mixed":
        precision = "16-mixed"
    else:
        precision = args.precision

    trainer = Trainer(
        logger=logger_tb,
        max_epochs=args.epochs,
        precision=precision,
        callbacks=[checkpoint_callback],
        accelerator='gpu',  # Assumes you have a GPU
        devices=1,  # Adjust based on your GPU availability
        
        
    )

    if args.progress:
        trainer.fit(model, ckpt_path=resume_ckpt)
    else:
        trainer.fit(model)

if __name__ == "__main__":
    main()
