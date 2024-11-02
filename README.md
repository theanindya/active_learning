Here is a `README.md` file to help users understand how to use the active learning pipeline in this repository.

# Active Learning Pipeline

This repository contains an active learning pipeline for training segmentation models. The pipeline supports various datasets and model architectures. This `README.md` provides instructions on how to set up, run, and use the active learning pipeline.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* Python 3.6 or higher
* PyTorch
* Albumentations
* Pytorch Lightning
* Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd active_learning
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training with Active Learning

To start training with active learning, use the `training_pipeline.py` script. Below are the command-line arguments you can use:

* `--ckpt`: Path where to store or load checkpoints.
* `--data_root`: Path to the dataset to use (Cityscapes, Mapillary, or SENSATION).
* `--data_type`: Specify the dataset to be used for training (`cityscapes`, `mapillary`, `sensation`).
* `--learning_rate`: Learning rate for fine-tuning.
* `--classes`: Number of classes to use for training.
* `--epochs`: Number of epochs to use for each active learning iteration.
* `--batch_size`: Number of batches to use during training.
* `--model_arc`: The model architecture to use during training.
* `--precision`: Precision for training (`16`, `32`, `64`, `bf16`, `mixed`).
* `--active_learning`: Enable active learning.
* `--al_iterations`: Number of active learning iterations.
* `--samples_per_iteration`: Number of samples to label in each active learning iteration.
* `--loss`: The loss function to use.
* `--early_stopping`: Enable early stopping.
* `--early_stopping_patience`: Number of epochs with no improvement after which training will be stopped.
* `--augmentation`: Comma-separated list of augmentations to apply (e.g., 'rotate,flip,scale').
* `--max_epochs`: Maximum number of epochs for training.
* `--min_epochs`: Minimum number of epochs for training.
* `--warmup_epochs`: Number of warmup epochs.
* `--use_ensemble`: Use ensemble for uncertainty estimation.
* `--initial_subset_size`: Initial size of the labeled subset.

Example command:
```sh
python training_pipeline.py --data_root /path/to/dataset --data_type cityscapes --learning_rate 1e-5 --classes 13 --epochs 20 --batch_size 4 --model_arc unet:timm-tf_efficientnet_lite0:8 --precision 32 --active_learning --al_iterations 5 --samples_per_iteration 100 --loss dice --early_stopping --early_stopping_patience 3 --augmentation rotate,flip,scale --max_epochs 50 --min_epochs 10 --warmup_epochs 5 --use_ensemble --initial_subset_size 100
```

### Evaluating the Model

To evaluate the model on a test set, use the `evaluate_model.py` script. Update the `checkpoint_path` and `test_dataset` variables in the script to point to your checkpoint and test dataset, respectively.

### Directory Structure

* `sensation/active_learning.py`: Contains the `ActiveLearningManager` class for managing the active learning process.
* `sensation/train/builder.py`: Contains functions for creating segmentation models and preparing datasets.
* `sensation/train/data.py`: Contains dataset classes for Cityscapes, Mapillary, and SENSATION datasets.
* `sensation/models/segmentation.py`: Contains segmentation model classes.
* `sensation/utils`: Contains utility functions for data processing, visualization, and analysis.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
