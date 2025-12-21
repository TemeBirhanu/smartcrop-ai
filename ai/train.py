"""
Main training script for SmartCrop AI
Usage: python train.py --config config/training.yaml --model mobilenet_v3
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import yaml
from src.training.train_classifier import train_classifier
from src.models.classifier_mobilenet import MobileNetV3Classifier
from src.models.classifier_effnet import EfficientNetClassifier
from src.data.dataset import CropDataset
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.helpers import get_device
from src.utils.file_utils import load_yaml, ensure_dir


def load_configs(base_config_path: str):
    """Load all configuration files."""
    config_dir = Path(base_config_path).parent
    
    configs = {
        'default': load_yaml(str(config_dir / 'default.yaml')),
        'model': load_yaml(str(config_dir / 'model.yaml')),
        'dataset': load_yaml(str(config_dir / 'dataset.yaml')),
        'training': load_yaml(base_config_path) if Path(base_config_path).exists() else {}
    }
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Train SmartCrop AI model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/training.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mobilenet_v3', 'efficientnet_b3'],
        default='mobilenet_v3',
        help='Model to train'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Path to data directory'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='data/splits',
        help='Path to annotations directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup logging
    log_dir = ensure_dir('outputs/logs')
    logger = setup_logger(
        log_file=str(log_dir / 'training.log'),
        level='INFO'
    )
    
    logger.info("=" * 60)
    logger.info("SmartCrop AI - Training Script")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Data dir: {args.data_dir}")
    
    # Load configurations
    try:
        configs = load_configs(args.config)
        logger.info("Configuration files loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load all configs: {e}. Using defaults.")
        configs = {
            'default': {'device': 'cuda', 'seed': 42},
            'model': {},
            'dataset': {'image_size': 224},
            'training': {}
        }
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model config
    model_config = configs['model'].get(args.model, {})
    num_classes = model_config.get('num_classes', 10)
    pretrained = model_config.get('pretrained', True)
    
    # Create model
    logger.info(f"Creating {args.model} model with {num_classes} classes...")
    if args.model == 'mobilenet_v3':
        model = MobileNetV3Classifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif args.model == 'efficientnet_b3':
        model = EfficientNetClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        from torch.utils.data import DataLoader
        from src.data.dataset import CropDataset
        
        image_size = configs['dataset'].get('image_size', 224)
        
        # Try to load datasets
        train_dataset = CropDataset(
            data_dir=os.path.join(args.data_dir, 'train'),
            annotations_file=os.path.join(args.annotations, 'train_annotations.csv'),
            image_size=image_size,
            augment=True
        )
        
        val_dataset = CropDataset(
            data_dir=os.path.join(args.data_dir, 'val'),
            annotations_file=os.path.join(args.annotations, 'val_annotations.csv'),
            image_size=image_size,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        logger.info("Note: You need to prepare your dataset first.")
        logger.info("Expected structure:")
        logger.info(f"  {args.data_dir}/train/ - training images")
        logger.info(f"  {args.data_dir}/val/ - validation images")
        logger.info(f"  {args.annotations}/train_annotations.csv - training annotations")
        logger.info(f"  {args.annotations}/val_annotations.csv - validation annotations")
        return
    
    # Run training
    try:
        logger.info("Starting training...")
        from src.training.train_classifier import train_classifier
        
        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=str(device),
            save_dir='outputs/models/checkpoints',
            freeze_backbone_layers=True
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

