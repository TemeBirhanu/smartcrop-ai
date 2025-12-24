"""
Intelligent dataset reduction script for transfer learning.

This script reduces dataset size while maintaining:
- Small classes intact (< 500 images)
- Large classes capped at reasonable limits
- Proportional train/val/test splits
- Random sampling for diversity
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def analyze_dataset(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Analyze current dataset structure and sizes."""
    stats = defaultdict(lambda: defaultdict(int))
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        for crop_dir in split_dir.iterdir():
            if not crop_dir.is_dir():
                continue
                
            for disease_dir in crop_dir.iterdir():
                if not disease_dir.is_dir():
                    continue
                
                # Count images
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                images = [
                    img for img in disease_dir.iterdir()
                    if img.suffix.lower() in image_extensions
                ]
                
                class_key = f"{crop_dir.name}/{disease_dir.name}"
                stats[class_key][split] = len(images)
    
    return dict(stats)


def get_reduction_targets(stats: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """
    Determine target sizes for each class.
    
    Strategy:
    - Classes with < 200 images: Keep all (too small to reduce)
    - Classes with 200-500 images: Reduce to 200
    - Classes with 500-1000 images: Reduce to 400
    - Classes with 1000-2000 images: Reduce to 600
    - Classes with > 2000 images: Reduce to 800
    """
    targets = {}
    
    for class_name, splits in stats.items():
        total = sum(splits.values())
        
        if total < 200:
            # Keep all - too small
            targets[class_name] = total
        elif total < 500:
            # Reduce to 200
            targets[class_name] = 200
        elif total < 1000:
            # Reduce to 400
            targets[class_name] = 400
        elif total < 2000:
            # Reduce to 600
            targets[class_name] = 600
        else:
            # Reduce to 800
            targets[class_name] = 800
    
    return targets


def reduce_class(
    data_dir: Path,
    crop: str,
    disease: str,
    target_total: int,
    current_stats: Dict[str, int]
) -> None:
    """
    Reduce images in a class across train/val/test splits.
    Maintains proportional distribution.
    """
    current_total = sum(current_stats.values())
    
    if current_total <= target_total:
        # No reduction needed
        return
    
    # Calculate target per split (maintain proportions)
    train_ratio = current_stats.get('train', 0) / current_total
    val_ratio = current_stats.get('val', 0) / current_total
    test_ratio = current_stats.get('test', 0) / current_total
    
    target_train = max(1, int(target_total * train_ratio))
    target_val = max(1, int(target_total * val_ratio))
    target_test = max(1, int(target_total * test_ratio))
    
    # Adjust if rounding causes overflow
    while target_train + target_val + target_test > target_total:
        if target_test > 1:
            target_test -= 1
        elif target_val > 1:
            target_val -= 1
        else:
            target_train -= 1
    
    # Process each split
    for split, target_count in [('train', target_train), ('val', target_val), ('test', target_test)]:
        split_dir = data_dir / split / crop / disease
        if not split_dir.exists():
            continue
        
        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            img for img in split_dir.iterdir()
            if img.suffix.lower() in image_extensions
        ]
        
        current_count = len(images)
        
        if current_count <= target_count:
            # No reduction needed for this split
            print(f"    {split}: {current_count} (keeping all)")
            continue
        
        # Randomly sample images to keep
        random.shuffle(images)
        images_to_keep = images[:target_count]
        images_to_remove = images[target_count:]
        
        # Remove excess images
        removed = 0
        for img_path in images_to_remove:
            try:
                img_path.unlink()
                removed += 1
            except Exception as e:
                print(f"      Warning: Could not delete {img_path}: {e}")
        
        print(f"    {split}: {current_count} -> {target_count} (removed {removed})")


def main():
    """Main reduction function."""
    print("=" * 60)
    print("Intelligent Dataset Reduction for Transfer Learning")
    print("=" * 60)
    print()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "ai" / "data" / "raw"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    print(f"Analyzing dataset in: {data_dir}")
    print()
    
    # Analyze current dataset
    print("Analyzing current dataset...")
    stats = analyze_dataset(data_dir)
    
    if not stats:
        print("No data found in dataset directory!")
        return
    
    # Calculate reduction targets
    targets = get_reduction_targets(stats)
    
    # Show analysis
    print("\n" + "=" * 60)
    print("Dataset Analysis & Reduction Plan")
    print("=" * 60)
    print()
    
    total_before = 0
    total_after = 0
    classes_to_reduce = []
    classes_to_keep = []
    
    for class_name in sorted(stats.keys()):
        current_total = sum(stats[class_name].values())
        target_total = targets[class_name]
        total_before += current_total
        total_after += target_total
        
        if current_total > target_total:
            classes_to_reduce.append((class_name, current_total, target_total))
        else:
            classes_to_keep.append((class_name, current_total))
    
    print(f"Total images before: {total_before:,}")
    print(f"Total images after:  {total_after:,}")
    print(f"Reduction: {((total_before - total_after) / total_before * 100):.1f}%")
    print()
    
    print(f"Classes to keep intact: {len(classes_to_keep)}")
    for class_name, count in classes_to_keep[:5]:
        print(f"  - {class_name}: {count} images")
    if len(classes_to_keep) > 5:
        print(f"  ... and {len(classes_to_keep) - 5} more")
    print()
    
    print(f"Classes to reduce: {len(classes_to_reduce)}")
    for class_name, before, after in classes_to_reduce[:10]:
        reduction = ((before - after) / before * 100)
        print(f"  - {class_name}: {before} -> {after} ({reduction:.0f}% reduction)")
    if len(classes_to_reduce) > 10:
        print(f"  ... and {len(classes_to_reduce) - 10} more")
    print()
    
    # Confirm before proceeding
    response = input("Proceed with reduction? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print()
    print("=" * 60)
    print("Reducing dataset...")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Reduce each class
    for class_name in sorted(stats.keys()):
        crop, disease = class_name.split('/', 1)
        current_total = sum(stats[class_name].values())
        target_total = targets[class_name]
        
        if current_total <= target_total:
            print(f"[SKIP] {class_name}: {current_total} (keeping all)")
            continue
        
        print(f"[REDUCE] {class_name}: {current_total} -> {target_total}")
        reduce_class(data_dir, crop, disease, target_total, stats[class_name])
    
    print()
    print("=" * 60)
    print("Reduction Complete!")
    print("=" * 60)
    print()
    
    # Final statistics
    final_stats = analyze_dataset(data_dir)
    final_total = sum(sum(splits.values()) for splits in final_stats.values())
    
    print(f"Final dataset size: {final_total:,} images")
    print(f"Reduced by: {total_before - final_total:,} images")
    print(f"Time saved: ~{((total_before - final_total) / 1000):.0f}k fewer images to process")
    print()
    print("Dataset is now optimized for faster transfer learning training!")


if __name__ == "__main__":
    main()

