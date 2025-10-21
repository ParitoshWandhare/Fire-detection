"""
Optimized training infrastructure - Target: F1 96%+, IoU 92%+

Key improvements:
1. Gradient accumulation for effective larger batch size
2. Warmup learning rate scheduler
3. Better memory management for 4GB GPU
4. Enhanced monitoring and logging
"""

import os
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import cv2

from .metrics import FireDetectionMetrics, MetricTracker


class WarmupCosineScheduler:
    """Cosine annealing with warmup for better convergence."""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, 
                 eta_min: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        # Ensure eta_min and base_lr are floats
        self.eta_min = float(eta_min)
        self.base_lr = float(optimizer.param_groups[0]['lr'])
        self.current_epoch = 0
    
    def step(self, epoch: int):
        self.current_epoch = int(epoch)
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            if self.total_epochs > self.warmup_epochs:
                progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            else:
                progress = 0.0
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr)
    
    def get_last_lr(self):
        return [float(param_group['lr']) for param_group in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping with best weights restoration."""
    
    def __init__(self, patience: int = 12, min_delta: float = 0.001,
                 mode: str = 'max', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_weights = None
    
    def __call__(self, current_value: float, epoch: int, model: nn.Module) -> bool:
        improved = False
        
        if self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                improved = True
        else:
            if current_value < self.best_value - self.min_delta:
                improved = True
        
        if improved:
            self.best_value = current_value
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                print(f"Restoring best weights from epoch {self.best_epoch}")
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device('cpu')
                model.load_state_dict({k: v.to(model_device) for k, v in self.best_weights.items()})
            return True
        
        return False


class LearningRateScheduler:
    """Wrapper supporting multiple schedulers including warmup."""
    
    def __init__(self, optimizer: optim.Optimizer, config: Dict):
        self.optimizer = optimizer
        self.config = config
        
        scheduler_type = config.get('type', config.get('name', 'cosine'))
        
        if scheduler_type == 'cosine_warmup':
            # Ensure all parameters are properly typed
            warmup_epochs = int(config.get('warmup_epochs', 5))
            total_epochs = int(config.get('T_max', 100))
            eta_min = float(config.get('eta_min', 1e-7))
            
            self.scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                eta_min=eta_min
            )
            self.scheduler_type = 'warmup'
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config.get('T_max', 100)),
                eta_min=float(config.get('eta_min', 1e-6))
            )
            self.scheduler_type = 'cosine'
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(config.get('step_size', 30)),
                gamma=float(config.get('gamma', 0.1))
            )
            self.scheduler_type = 'step'
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get('mode', 'max'),
                factor=float(config.get('factor', 0.5)),
                patience=int(config.get('patience', 5)),
                threshold=float(config.get('threshold', 0.001))
            )
            self.scheduler_type = 'plateau'
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric_value: Optional[float] = None, epoch: Optional[int] = None):
        if self.scheduler_type == 'warmup' and epoch is not None:
            self.scheduler.step(int(epoch))
        elif self.scheduler_type == 'plateau':
            if metric_value is not None:
                self.scheduler.step(float(metric_value))
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> float:
        try:
            lr_list = self.scheduler.get_last_lr()
            return float(lr_list[0]) if isinstance(lr_list, list) else float(lr_list)
        except Exception:
            return float(self.optimizer.param_groups[0]['lr'])


class FireDetectionTrainer:
    """Optimized trainer for fire detection - Target: F1 96%+, IoU 92%+"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: nn.Module,
                 optimizer: optim.Optimizer,
                 config: Dict,
                 device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training settings
        self.epochs = config.get('epochs', 100)
        self.grad_clip = config.get('gradient_clipping', 1.0)
        self.mixed_precision = config.get('mixed_precision', True)
        self.log_interval = config.get('log_interval', 10)
        self.val_interval = config.get('validation_interval', 1)
        self.save_interval = config.get('save_interval', 5)
        
        # Gradient accumulation for effective larger batch size
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Memory optimization for 4GB GPU
        self.empty_cache_freq = config.get('memory_optimization', {}).get('empty_cache_every_n_steps', 50)
        
        # Components
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Scheduler with warmup support
        if 'scheduler' in config:
            self.scheduler = LearningRateScheduler(optimizer, config['scheduler'])
        else:
            self.scheduler = None
        
        # Early stopping
        if 'early_stopping' in config:
            es_config = config['early_stopping']
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 12),
                min_delta=es_config.get('min_delta', 0.001),
                mode=es_config.get('mode', 'max'),
                restore_best_weights=es_config.get('restore_best_weights', True)
            )
        else:
            self.early_stopping = None
        
        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        
        # State
        self.current_epoch = 0
        self.best_val_metric = float('-inf')
        self.training_history = []
        
        # Logging
        self.setup_logging()
        self._validate_config()
        self._sanity_done = False
    
    def setup_logging(self):
        log_file = self.log_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Optimized trainer initialized")
    
    def _denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            img = tensor.detach().cpu().float()
        else:
            img = torch.tensor(tensor).float()
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        img = (img * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return img
    
    def _save_overlay(self, image_tensor: torch.Tensor, mask_tensor: torch.Tensor,
                      pred_tensor: Optional[torch.Tensor], save_path: Union[str, Path]):
        img = self._denormalize_image(image_tensor)
        
        mask = mask_tensor.detach().cpu().numpy()
        if mask.ndim > 2:
            mask = np.squeeze(mask)
            if mask.ndim > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask
        mask = mask.astype(np.uint8)
        
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if pred_tensor is not None:
            pred = pred_tensor.detach().cpu().numpy()
            if pred.ndim > 2:
                pred = np.squeeze(pred)
                if pred.ndim > 2:
                    pred = pred[0] if pred.shape[0] == 1 else pred
            pred = pred.astype(np.uint8)
            
            if pred.shape != img.shape[:2]:
                pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            pred = np.zeros_like(mask, dtype=np.uint8)
        
        overlay = img.copy()
        
        mask_bool = (mask == 1)
        if np.any(mask_bool):
            overlay[mask_bool] = (overlay[mask_bool] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        
        pred_bool = (pred == 1)
        if np.any(pred_bool):
            overlay[pred_bool] = (overlay[pred_bool] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
        
        os.makedirs(Path(save_path).parent, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    def _run_data_sanity_checks(self, num_examples: int = 2):
        if self._sanity_done:
            return
        self.logger.info("Running data sanity checks")
        
        def sample_from_loader(loader, split_name: str):
            try:
                batch = next(iter(loader))
            except Exception as e:
                self.logger.warning(f"Could not sample {split_name} loader: {e}")
                return
            
            images, masks = batch
            for i in range(min(num_examples, images.shape[0])):
                img = images[i]
                mask = masks[i]
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                mask = mask.long()
                
                pos = int((mask == 1).sum().item())
                total = mask.numel()
                pct = pos / total * 100.0
                self.logger.info(f"{split_name} sample {i}: fire pixels = {pos} ({pct:.4f}%)")
                
                save_name = self.log_dir / 'sanity_samples' / f"{split_name}_sample_{i}.png"
                self._save_overlay(img, mask, None, save_name)
        
        sample_from_loader(self.train_loader, 'train')
        sample_from_loader(self.val_loader, 'val')
        self._sanity_done = True
    
    def save_checkpoint(self, epoch: int, is_best: bool = False,
                       additional_info: Optional[Dict] = None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': dict(self.train_metrics.metrics_history),
            'val_metrics': dict(self.val_metrics.metrics_history),
            'config': self.config,
            'best_val_metric': self.best_val_metric
        }
        
        if self.scheduler:
            try:
                checkpoint['scheduler_state_dict'] = self.scheduler.scheduler.state_dict()
            except Exception:
                pass
        
        if self.scaler:
            try:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            except Exception:
                pass
        
        if additional_info:
            checkpoint.update(additional_info)
        
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at epoch {epoch}")
        
        if epoch % self.save_interval == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
        
        self.logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path],
                       load_optimizer: bool = True, load_scheduler: bool = True):
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                self.logger.warning(f"Could not load optimizer state: {e}")
        
        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                self.logger.warning(f"Could not load scheduler state: {e}")
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                self.logger.warning(f"Could not load scaler state: {e}")
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))
        
        if 'train_metrics' in checkpoint:
            for key, values in checkpoint['train_metrics'].items():
                self.train_metrics.metrics_history[key] = values
        
        if 'val_metrics' in checkpoint:
            for key, values in checkpoint['val_metrics'].items():
                self.val_metrics.metrics_history[key] = values
        
        self.logger.info(f"Resumed training from epoch {self.current_epoch}")
        return checkpoint
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        
        train_fire_metrics = FireDetectionMetrics(device=self.device)
        epoch_losses = defaultdict(list)
        
        total_batches = len(self.train_loader) if len(self.train_loader) > 0 else 0
        start_time = time.time()
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            if self.mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, masks)
                    total_loss = loss_dict['loss'] / self.grad_accum_steps
                
                self.scaler.scale(total_loss).backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss_dict = self.loss_fn(outputs, masks)
                total_loss = loss_dict['loss'] / self.grad_accum_steps
                
                total_loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            with torch.no_grad():
                if isinstance(outputs, dict) and 'out' in outputs:
                    preds_for_metrics = outputs['out']
                else:
                    preds_for_metrics = outputs
                train_fire_metrics.update(preds_for_metrics, masks, probs=preds_for_metrics)
            
            # Accumulate losses
            for key, value in loss_dict.items():
                try:
                    epoch_losses[key].append(float(value.item()) * self.grad_accum_steps)
                except Exception:
                    try:
                        epoch_losses[key].append(float(value) * self.grad_accum_steps)
                    except Exception:
                        pass
            
            # Memory management for 4GB GPU
            if (batch_idx + 1) % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()
            
            # Log progress
            if total_batches > 0 and (batch_idx % self.log_interval == 0):
                progress = 100.0 * batch_idx / total_batches
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1) if (batch_idx + 1) > 0 else 0.0
                
                self.logger.info(
                    f'Train Epoch {self.current_epoch} [{batch_idx:5d}/{total_batches}] '
                    f'({progress:3.0f}%) | Loss: {float(total_loss.item() * self.grad_accum_steps if hasattr(total_loss, "item") else total_loss):.4f} | '
                    f'ETA: {eta/60:.1f}min'
                )
        
        # Compute epoch metrics
        train_metrics = train_fire_metrics.compute_all_metrics()
        
        for key, losses in epoch_losses.items():
            if len(losses) > 0:
                train_metrics[f'train_{key}'] = float(np.mean(losses))
        
        if self.scheduler:
            train_metrics['learning_rate'] = self.scheduler.get_last_lr()
        else:
            train_metrics['learning_rate'] = float(self.optimizer.param_groups[0]['lr'])
        
        epoch_time = time.time() - start_time
        train_metrics['epoch_time'] = epoch_time
        
        self.logger.info(f'Train Epoch {self.current_epoch} completed in {epoch_time/60:.1f}min')
        
        return train_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        val_fire_metrics = FireDetectionMetrics(device=self.device)
        epoch_losses = defaultdict(list)
        
        total_batches = len(self.val_loader) if len(self.val_loader) > 0 else 0
        start_time = time.time()
        
        samples_saved = False
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                if self.mixed_precision and self.scaler is not None:
                    with autocast():
                        outputs = self.model(images)
                        loss_dict = self.loss_fn(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, masks)
                
                # Update metrics
                if isinstance(outputs, dict) and 'out' in outputs:
                    preds_for_metrics = outputs['out']
                else:
                    preds_for_metrics = outputs
                val_fire_metrics.update(preds_for_metrics, masks, probs=preds_for_metrics)
                
                for key, value in loss_dict.items():
                    try:
                        epoch_losses[key].append(float(value.item()))
                    except Exception:
                        try:
                            epoch_losses[key].append(float(value))
                        except Exception:
                            pass
                
                # Save samples
                if not samples_saved:
                    try:
                        probs = preds_for_metrics
                        if probs.dim() == 4 and probs.shape[1] == 1:
                            probs_map = torch.sigmoid(probs.squeeze(1))
                        elif probs.dim() == 4:
                            probs_map = torch.softmax(probs, dim=1)[:, 1]
                        else:
                            probs_map = probs
                        
                        preds_bin = (probs_map > 0.45).long()  # Use optimized threshold
                        
                        for i in range(min(4, images.shape[0])):
                            img = images[i].cpu()
                            gt_mask = masks[i].cpu()
                            pred_mask = preds_bin[i].cpu()
                            save_path = self.log_dir / 'validation_samples' / f'epoch{self.current_epoch}_batch{batch_idx}_sample{i}.png'
                            self._save_overlay(img, gt_mask, pred_mask, save_path)
                        samples_saved = True
                    except Exception as e:
                        self.logger.warning(f"Failed to save validation samples: {e}")
        
        val_metrics = val_fire_metrics.compute_all_metrics()
        
        for key, losses in epoch_losses.items():
            if len(losses) > 0:
                val_metrics[f'val_{key}'] = float(np.mean(losses))
        
        epoch_time = time.time() - start_time
        val_metrics['val_epoch_time'] = epoch_time
        
        if getattr(val_fire_metrics, 'fire_tp', 0) == 0 and getattr(val_fire_metrics, 'fire_fp', 0) == 0:
            self.logger.warning("No positive predictions in validation. Check model/data.")
        
        self.logger.info(f'Validation completed in {epoch_time/60:.1f}min')
        
        return val_metrics
    
    def log_metrics(self, train_metrics: Dict[str, float],
                   val_metrics: Optional[Dict[str, float]] = None):
        key_train_metrics = ['train_loss', 'fire_f1', 'mean_iou', 'fire_detection_rate']
        train_msg = " | ".join([f"{k}: {train_metrics.get(k, 0):.4f}" for k in key_train_metrics])
        
        if val_metrics:
            val_display_metrics = {
                'val_loss': val_metrics.get('val_loss', 0),
                'val_fire_f1': val_metrics.get('fire_f1', 0),
                'val_mean_iou': val_metrics.get('mean_iou', 0),
                'val_fire_detection_rate': val_metrics.get('fire_detection_rate', 0)
            }
            
            val_msg = " | ".join([f"{k}: {v:.4f}" for k, v in val_display_metrics.items()])
            
            self.logger.info(f'Epoch {self.current_epoch} - Train: {train_msg}')
            self.logger.info(f'Epoch {self.current_epoch} - Val: {val_msg}')
        else:
            self.logger.info(f'Epoch {self.current_epoch} - Train: {train_msg}')
    
    def train(self, resume_checkpoint: Optional[str] = None) -> Dict:
        """Main training loop with all optimizations."""
        
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
        
        self.logger.info("Starting optimized training...")
        self.logger.info(f"Training for {self.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.mixed_precision}")
        self.logger.info(f"Gradient Accumulation Steps: {self.grad_accum_steps}")
        
        try:
            self._run_data_sanity_checks()
        except Exception as e:
            self.logger.warning(f"Data sanity checks failed: {e}")
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Training
                train_metrics = self.train_epoch()
                self.train_metrics.update(train_metrics)
                
                # Validation
                val_metrics = None
                if epoch % self.val_interval == 0:
                    val_metrics = self.validate_epoch()
                    self.val_metrics.update(val_metrics)
                
                # Log metrics
                self.log_metrics(train_metrics, val_metrics)
                
                # Learning rate scheduling
                if self.scheduler:
                    if hasattr(self.scheduler, 'scheduler_type') and self.scheduler.scheduler_type == 'warmup':
                        self.scheduler.step(epoch=epoch)
                    elif val_metrics:
                        monitor_metric = val_metrics.get('fire_f1', val_metrics.get('mean_iou', 0))
                        self.scheduler.step(monitor_metric, epoch)
                    else:
                        self.scheduler.step(epoch=epoch)
                
                # Check for best model
                is_best = False
                if val_metrics:
                    current_metric = val_metrics.get('fire_f1', val_metrics.get('mean_iou', 0))
                    if current_metric > self.best_val_metric:
                        self.best_val_metric = current_metric
                        is_best = True
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping
                if self.early_stopping and val_metrics:
                    monitor_metric = val_metrics.get('fire_f1', val_metrics.get('mean_iou', 0))
                    if self.early_stopping(monitor_metric, epoch, self.model):
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                # Update history
                epoch_history = {
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'epoch_time': time.time() - epoch_start_time
                }
                self.training_history.append(epoch_history)
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            total_training_time = time.time() - training_start_time
            self.logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
            
            history_path = self.log_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)
        
        return {
            'training_history': self.training_history,
            'best_val_metric': self.best_val_metric,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_training_time
        }
    
    def _validate_config(self):
        required_fields = ['epochs', 'optimizer', 'loss']
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")


def create_trainer(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                  loss_fn: nn.Module, optimizer: optim.Optimizer, config: Dict,
                  device: str = 'cuda', checkpoint_dir: str = 'checkpoints',
                  log_dir: str = 'logs') -> FireDetectionTrainer:
    """Factory function to create optimized trainer."""
    return FireDetectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )


if __name__ == "__main__":
    print("Optimized Fire Detection Training Infrastructure")
    print("=" * 50)
    print("✅ Target: F1 96%+, IoU 92%+")
    print("✅ Optimized for GTX 1650 (4GB)")