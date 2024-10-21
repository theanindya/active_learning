import os
import torch.nn as nn
import re
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from sensation.train import builder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

class ActiveLearningManager:
    def __init__(self, checkpoint_dir, train_dataset, val_dataset, args):
        self.checkpoint_dir = checkpoint_dir
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        print(f"Validation set size: {len(self.val_dataset)}")
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ActiveLearningManager initialized on: {self.device}")
        self.best_checkpoint, self.best_original_iou = self.find_best_checkpoint()
        self.model = self.load_model_from_checkpoint(self.best_checkpoint)
        self.model = self.model.to(self.device)
        self.logger = TensorBoardLogger("tb_logs", name="active_learning")
        self.max_epochs = getattr(args, 'max_epochs', 50)  # default to 50 if not provided
        self.min_epochs = getattr(args, 'min_epochs', 10)  # default to 10 if not provided
        self.warmup_epochs = getattr(args, 'warmup_epochs', 5)  # default to 5 if not provided
        self.use_ensemble = getattr(args, 'use_ensemble', False)
        self.weight_decay = getattr(args, 'weight_decay', 0)
    
        if self.use_ensemble:
            self.ensemble = self.create_ensemble(2)
        else:
            self.ensemble = None

        self.all_indices = list(range(len(self.train_dataset)))
        self.class_counts = self.get_class_counts()


    def find_best_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.ckpt') and not f.startswith('last')]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {self.checkpoint_dir}")
        best_iou = 0
        best_checkpoint = None
        for ckpt in checkpoints:
            match = re.search(r'val_iou=(\d+\.\d+)', ckpt)
            if match:
                iou = float(match.group(1))
                if iou > best_iou:
                    best_iou = iou
                    best_checkpoint = os.path.join(self.checkpoint_dir, ckpt)
        if best_checkpoint is None:
            raise ValueError("Could not find a valid checkpoint with IoU information")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Best original IoU: {best_iou}")
        return best_checkpoint, best_iou

    def create_ensemble(self, num_models):
        ensemble = []
        for _ in range(num_models):
            model = self.load_model_from_checkpoint(self.best_checkpoint)
            model = model.to(self.device)
            model.eval()
            ensemble.append(model)
        return ensemble

    def to_device(self, model):
        return model.to(self.device)

    def load_model_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        num_classes = state_dict['whole_model.arc.segmentation_head.0.weight'].size(0)
        self.args.classes = num_classes
        model = builder.create_seg_model(
            model_arc=self.args.model_arc,
            epochs=self.args.epochs,
            num_classes=num_classes,
            learning_rate=self.args.learning_rate,
            batch_size=self.args.batch_size,
            loss=builder.get_loss(self.args.loss, num_classes=self.args.classes),
            train_data=None,
            val_data=None,
            test_data=None,
        )
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        print(f"Loaded model device: {next(model.parameters()).device}")
        return model

    def get_uncertainty_scores(self, dataloader):
        if self.use_ensemble:
            return self.get_ensemble_uncertainty(dataloader)
        else:
            return self.get_single_model_uncertainty(dataloader)

    def get_ensemble_uncertainty(self, dataloader):
        uncertainties = []
        smaller_batch_size = self.args.batch_size // 2
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                for i in range(0, inputs.size(0), smaller_batch_size):
                    batch_slice = inputs[i:i+smaller_batch_size]
                    ensemble_outputs = [model(batch_slice) for model in self.ensemble]
                    ensemble_probs = [torch.nn.functional.softmax(output, dim=1) for output in ensemble_outputs]
                    mean_probs = torch.mean(torch.stack(ensemble_probs), dim=0)
                    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-5), dim=1)
                    uncertainties.extend(entropy.cpu().numpy().mean(axis=(1, 2)))
        return np.array(uncertainties)

    def select_samples(self, unlabeled_indices, num_samples):
        unlabeled_subset = Subset(self.train_dataset, unlabeled_indices)
        unlabeled_loader = DataLoader(unlabeled_subset, batch_size=self.args.batch_size, shuffle=False)
        uncertainties = self.get_uncertainty_scores(unlabeled_loader)
        try:
            features = self.get_feature_representations(unlabeled_loader)
            uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
            features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
            combined = np.column_stack((uncertainties.reshape(-1, 1), features))
            kmeans = KMeans(n_clusters=num_samples, random_state=42)
            kmeans.fit(combined)
            selected_indices = []
            for i in range(num_samples):
                cluster_samples = np.where(kmeans.labels_ == i)[0]
                most_uncertain = cluster_samples[uncertainties[cluster_samples].argmax()]
                selected_indices.append(most_uncertain)
        except Exception as e:
            print(f"Error in feature extraction or clustering: {e}")
            print("Falling back to uncertainty-based selection")
            selected_indices = np.argsort(uncertainties)[-num_samples:]
        new_labeled_indices = [unlabeled_indices[i] for i in selected_indices]
        return new_labeled_indices

    def get_feature_representations(self, dataloader):
        features = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                encoder = self.model.whole_model.encoder if hasattr(self.model.whole_model, 'encoder') else self.model.whole_model
                if isinstance(encoder, nn.Sequential):
                    for layer in encoder:
                        inputs = layer(inputs)
                        if isinstance(layer, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                            break
                else:
                    inputs = encoder(inputs)
                batch_features = inputs.view(inputs.size(0), -1)
                features.append(batch_features.cpu().numpy())
        return np.concatenate(features, axis=0)

    def update_model(self, labeled_indices, epochs):
        labeled_subset = Subset(self.train_dataset, labeled_indices)
        self.model = builder.create_seg_model(
            model_arc=self.args.model_arc,
            epochs=epochs,
            num_classes=self.args.classes,
            learning_rate=self.args.learning_rate,
            batch_size=self.args.batch_size,
            loss=builder.get_loss(self.args.loss),
            train_data=labeled_subset,
            val_data=self.val_dataset,
            test_data=None,
        )
        self.model = self.to_device(self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        early_stop_callback = EarlyStopping(
            monitor='val_iou',
            min_delta=0.00,
            patience=self.args.early_stopping_patience,
            verbose=True,
            mode='max'
        )
        trainer = Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            devices=1,
            callbacks=[self.args.checkpoint_callback, early_stop_callback],
            logger=self.logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
            precision="16-mixed",
        )
        trainer.fit(self.model)
        if self.use_ensemble:
            self.update_ensemble()
        return trainer.callback_metrics.get('val_iou', 0).item()

    def update_ensemble(self):
        self.ensemble = self.ensemble[1:] + [self.to_device(self.model)]

    def get_class_counts(self):
        class_counts = Counter()
        for idx in self.all_indices:
            _, mask = self.train_dataset[idx]
            classes_in_image = np.unique(mask)
            class_counts.update(classes_in_image)
        return class_counts

    def get_balanced_subset(self, num_samples, exclude=[]):
        available_indices = list(set(self.all_indices) - set(exclude))
        class_weights = {cls: 1.0/count for cls, count in self.class_counts.items() if cls != 0}
        total_weight = sum(class_weights.values())
        class_weights = {cls: weight/total_weight for cls, weight in class_weights.items()}
        selected_indices = []
        while len(selected_indices) < num_samples and available_indices:
            idx = np.random.choice(available_indices)
            _, mask = self.train_dataset[idx]
            classes_in_image = np.unique(mask)
            if any(cls in class_weights for cls in classes_in_image):
                selected_indices.append(idx)
                available_indices.remove(idx)
        return selected_indices

    def run_active_learning(self, initial_subset_size, max_iterations, samples_per_iteration, epochs_per_iteration):
        current_indices = self.get_balanced_subset(initial_subset_size)
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"Curriculum Learning Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*50}")
            print(f"Current training set size: {len(current_indices)}")
            self.update_model(current_indices, epochs_per_iteration)
            if len(current_indices) >= len(self.all_indices):
                print("All samples are now included. Stopping the learning process.")
                break
            new_indices = self.get_balanced_subset(samples_per_iteration, exclude=current_indices)
            current_indices.extend(new_indices)
        return self.model
