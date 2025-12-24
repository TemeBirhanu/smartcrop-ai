"""
Training script for fine-tuning pretrained classifiers
Only trains classification head - backbone stays frozen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from typing import Dict, Optional
import yaml

from ..models.model_utils import freeze_backbone
from .loss_functions import CombinedLoss
from .metrics import AverageMeter, calculate_metrics
from .scheduler import get_scheduler


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,  # Few epochs needed with pretrained weights
    learning_rate: float = 0.001,
    device: str = "cuda",
    save_dir: str = "outputs/models",
    freeze_backbone_layers: bool = True,  # Keep pretrained weights frozen
    resume_from: Optional[str] = None
) -> Dict:
    """
    Train classifier by fine-tuning pretrained model.
    
    Strategy:
    1. Freeze backbone (pretrained weights stay unchanged)
    2. Train only classification head
    3. Requires minimal data and epochs
    
    Args:
        model: PyTorch model (with pretrained backbone)
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs (5-10 usually enough)
        learning_rate: Learning rate for optimizer
        device: Device to train on
        save_dir: Directory to save checkpoints
        freeze_backbone_layers: Freeze backbone (default: True)
        resume_from: Path to checkpoint to resume from
    
    Returns:
        Training history dictionary
    """
    # Move model to device
    model = model.to(device)
    
    # Freeze backbone - keep pretrained weights unchanged
    if freeze_backbone_layers:
        model = freeze_backbone(model, freeze=True)
        print("✓ Backbone frozen - only classification head will be trained")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Loss function
    criterion = CombinedLoss(use_focal=False)
    
    # Optimizer - only optimize trainable parameters (classification head)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Learning rate scheduler
    scheduler = get_scheduler(optimizer, num_epochs)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume_from and os.path.exists(resume_from):
        # Use weights_only=False for PyTorch 2.6+ compatibility (we trust our own checkpoints)
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train phase
        model.train()
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        
        train_pbar = tqdm(train_loader, desc="Training")
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == labels).float().mean()
            
            train_loss_meter.update(loss.item(), images.size(0))
            train_acc_meter.update(acc.item(), images.size(0))
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            # Get number of classes from model
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'classifier'):
                if isinstance(model.backbone.classifier, nn.Sequential):
                    num_classes = model.backbone.classifier[-1].out_features
                else:
                    num_classes = model.backbone.classifier.out_features
            else:
                # Fallback: get from last layer
                for module in model.modules():
                    if isinstance(module, nn.Linear):
                        num_classes = module.out_features
            
            val_pbar = tqdm(val_loader, desc="Validation")
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Safety check: ensure labels are in valid range
                if labels.max() >= num_classes or labels.min() < 0:
                    print(f"⚠️  Invalid labels detected: min={labels.min()}, max={labels.max()}, num_classes={num_classes}")
                    print(f"   This batch will be skipped. Check dataset class mapping.")
                    continue
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                acc = (predicted == labels).float().mean()
                
                val_loss_meter.update(loss.item(), images.size(0))
                val_acc_meter.update(acc.item(), images.size(0))
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc.item():.4f}'
                })
        
        # Calculate detailed metrics
        metrics = calculate_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=len(torch.unique(torch.tensor(all_labels)))
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"Train Loss: {train_loss_meter.avg:.4f}, Train Acc: {train_acc_meter.avg:.4f}")
        print(f"Val Loss: {val_loss_meter.avg:.4f}, Val Acc: {val_acc_meter.avg:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}, LR: {current_lr:.6f}")
        
        # Save history
        history['train_loss'].append(train_loss_meter.avg)
        history['train_acc'].append(train_acc_meter.avg)
        history['val_loss'].append(val_loss_meter.avg)
        history['val_acc'].append(val_acc_meter.avg)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'val_acc': val_acc_meter.avg,
            'metrics': metrics
        }
        
        # Save best model
        if val_acc_meter.avg > best_val_acc:
            best_val_acc = val_acc_meter.avg
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Saved best model (Val Acc: {best_val_acc:.4f})")
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return history

