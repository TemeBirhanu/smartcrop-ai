"""
Model export script for SmartCrop AI
Usage: python export_model.py --model mobilenet_v3 --checkpoint outputs/models/checkpoints/mobilenet_v3_best.pth
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.export.export_utils import export_model
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description='Export SmartCrop AI model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['mobilenet_v3', 'efficientnet_b3'],
        default='mobilenet_v3',
        help='Model to export'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/export.yaml',
        help='Path to export config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for export (cpu recommended)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(level='INFO')
    
    logger.info("=" * 60)
    logger.info("SmartCrop AI - Model Export Script")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    if args.checkpoint:
        logger.info(f"Checkpoint: {args.checkpoint}")
    
    try:
        # Export model
        export_paths = export_model(
            config_path=args.config,
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("EXPORT COMPLETE")
        logger.info("=" * 60)
        for format_type, path in export_paths.items():
            logger.info(f"{format_type.upper()}: {path}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()


