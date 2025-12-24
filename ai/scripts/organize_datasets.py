#!/usr/bin/env python3
"""
Script to organize PlantVillage and Wheat datasets into the expected structure.

Expected structure:
data/raw/
├── wheat/
│   ├── healthy/
│   ├── septoria/
│   └── stripe_rust/
├── maize/
│   ├── healthy/
│   ├── common_rust/
│   └── ...
└── ...
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import random

def parse_plantvillage_folder(folder_name: str) -> Tuple[str, str]:
    """
    Parse PlantVillage folder name: "Crop___Disease" -> (crop, disease)
    
    Examples:
    - "Corn_(maize)___Common_rust_" -> ("maize", "common_rust")
    - "Apple___Apple_scab" -> ("apple", "apple_scab")
    - "Cherry_(including_sour)___healthy" -> ("cherry", "healthy")
    """
    if "___" not in folder_name:
        return None, None
    
    parts = folder_name.split("___")
    crop_part = parts[0].strip()
    disease_part = parts[1].strip() if len(parts) > 1 else "healthy"
    
    # Normalize crop names
    crop_mapping = {
        "Corn_(maize)": "maize",
        "Corn": "maize",
        "Cherry_(including_sour)": "cherry",
        "Cherry": "cherry",
        "Pepper,_bell": "pepper",
        "Pepper_bell": "pepper",
        "Pepper": "pepper",
    }
    
    # Clean crop name
    crop = crop_part.replace("_", " ").strip().lower()
    crop = crop_mapping.get(crop_part, crop)
    crop = crop.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    
    # Normalize disease names
    disease = disease_part.replace("_", " ").strip().lower()
    disease = disease.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    
    # Remove trailing underscores
    crop = crop.rstrip("_")
    disease = disease.rstrip("_")
    
    return crop, disease


def organize_plantvillage(
    source_dir: Path,
    target_dir: Path,
    split: str = "train"
) -> dict:
    """
    Organize PlantVillage dataset from Crop___Disease format to crop/disease format.
    
    Args:
        source_dir: Source directory (e.g., data/row/train/)
        target_dir: Target directory (e.g., data/raw/)
        split: "train", "val", or "all"
    
    Returns:
        Dictionary with organization statistics
    """
    stats = {
        "crops": set(),
        "diseases": set(),
        "images_moved": 0,
        "errors": []
    }
    
    if not source_dir.exists():
        print(f"[WARNING] Source directory not found: {source_dir}")
        return stats
    
    # Get all Crop___Disease folders
    folders = [d for d in source_dir.iterdir() if d.is_dir()]
    
    print(f"\nProcessing {len(folders)} folders from {source_dir.name}...")
    
    for folder in folders:
        crop, disease = parse_plantvillage_folder(folder.name)
        
        if not crop or not disease:
            print(f"[WARNING] Skipping folder (couldn't parse): {folder.name}")
            stats["errors"].append(folder.name)
            continue
        
        # Create target directory: data/raw/crop/disease/
        target_crop_dir = target_dir / crop
        target_disease_dir = target_crop_dir / disease
        target_disease_dir.mkdir(parents=True, exist_ok=True)
        
        # Move/copy images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in folder.iterdir() if f.suffix in image_extensions]
        
        for img in images:
            # Create unique filename if needed
            target_path = target_disease_dir / img.name
            if target_path.exists():
                # Add prefix to avoid overwriting
                target_path = target_disease_dir / f"{folder.name}_{img.name}"
            
            try:
                shutil.copy2(img, target_path)
                stats["images_moved"] += 1
            except Exception as e:
                stats["errors"].append(f"Error copying {img}: {e}")
        
        stats["crops"].add(crop)
        stats["diseases"].add(f"{crop}/{disease}")
        
        print(f"  [OK] {crop}/{disease}: {len(images)} images")
    
    return stats


def organize_wheat_dataset(
    source_dir: Path,
    target_dir: Path
) -> dict:
    """
    Organize Wheat Leaf dataset to match expected structure.
    
    Expected source: data/row/wheat_leaf/Healthy/, septoria/, stripe_rust/
    Target: data/raw/wheat/healthy/, septoria/, stripe_rust/
    """
    stats = {
        "diseases": set(),
        "images_moved": 0,
        "errors": []
    }
    
    if not source_dir.exists():
        print(f"[WARNING] Wheat dataset directory not found: {source_dir}")
        return stats
    
    print(f"\nOrganizing Wheat dataset from {source_dir}...")
    
    # Create wheat directory
    wheat_dir = target_dir / "wheat"
    wheat_dir.mkdir(parents=True, exist_ok=True)
    
    # Disease name mapping
    disease_mapping = {
        "Healthy": "healthy",
        "healthy": "healthy",
        "septoria": "septoria",
        "stripe_rust": "stripe_rust",
        "stripe rust": "stripe_rust",
        "Stripe_rust": "stripe_rust",
    }
    
    # Process each disease folder
    for disease_folder in source_dir.iterdir():
        if not disease_folder.is_dir():
            continue
        
        # Normalize disease name
        disease_name = disease_folder.name
        normalized_disease = disease_mapping.get(disease_name, disease_name.lower().replace(" ", "_"))
        
        # Create target directory
        target_disease_dir = wheat_dir / normalized_disease
        target_disease_dir.mkdir(parents=True, exist_ok=True)
        
        # Move/copy images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in disease_folder.iterdir() if f.suffix in image_extensions]
        
        for img in images:
            target_path = target_disease_dir / img.name
            if target_path.exists():
                target_path = target_disease_dir / f"{disease_folder.name}_{img.name}"
            
            try:
                shutil.copy2(img, target_path)
                stats["images_moved"] += 1
            except Exception as e:
                stats["errors"].append(f"Error copying {img}: {e}")
        
        stats["diseases"].add(normalized_disease)
        print(f"  [OK] wheat/{normalized_disease}: {len(images)} images")
    
    return stats


def create_train_val_split(
    data_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    """
    Create train/val/test splits from organized dataset.
    
    This creates separate directories for train/val/test splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"
    
    print(f"\nCreating train/val/test splits...")
    
    # Create split directories
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(exist_ok=True)
    
    # Process each crop
    for crop_dir in data_dir.iterdir():
        if not crop_dir.is_dir() or crop_dir.name in ["train", "val", "test"]:
            continue
        
        crop_name = crop_dir.name
        
        # Create crop directories in each split
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / crop_name).mkdir(exist_ok=True)
        
        # Process each disease
        for disease_dir in crop_dir.iterdir():
            if not disease_dir.is_dir():
                continue
            
            disease_name = disease_dir.name
            
            # Get all images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
            images = [f for f in disease_dir.iterdir() if f.suffix in image_extensions]
            
            if len(images) == 0:
                continue
            
            # Shuffle and split
            random.shuffle(images)
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Create disease directories in each split
            for split_dir, split_images in [
                (train_dir, train_images),
                (val_dir, val_images),
                (test_dir, test_images)
            ]:
                target_disease_dir = split_dir / crop_name / disease_name
                target_disease_dir.mkdir(exist_ok=True)
                
                for img in split_images:
                    shutil.copy2(img, target_disease_dir / img.name)
            
            print(f"  [OK] {crop_name}/{disease_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")


def main():
    """Main function to organize datasets."""
    print("=" * 60)
    print("Dataset Organization Script")
    print("=" * 60)
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    source_row_dir = base_dir / "data" / "row"
    target_raw_dir = base_dir / "data" / "raw"
    
    # Create target directory
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if source directory exists
    if not source_row_dir.exists():
        print(f"\n[ERROR] Source directory not found: {source_row_dir}")
        print("Please ensure your datasets are in data/row/")
        return
    
    # Organize PlantVillage train set
    plantvillage_train = source_row_dir / "train"
    if plantvillage_train.exists():
        print("\n" + "=" * 60)
        print("Organizing PlantVillage TRAIN set...")
        print("=" * 60)
        stats_train = organize_plantvillage(plantvillage_train, target_raw_dir, split="train")
        print(f"\n[OK] Train: {stats_train['images_moved']} images, {len(stats_train['crops'])} crops")
    
    # Organize PlantVillage val set
    plantvillage_val = source_row_dir / "val"
    if plantvillage_val.exists():
        print("\n" + "=" * 60)
        print("Organizing PlantVillage VAL set...")
        print("=" * 60)
        stats_val = organize_plantvillage(plantvillage_val, target_raw_dir, split="val")
        print(f"\n[OK] Val: {stats_val['images_moved']} images, {len(stats_val['crops'])} crops")
    
    # Organize Wheat dataset
    wheat_source = source_row_dir / "wheat_leaf"
    if wheat_source.exists():
        print("\n" + "=" * 60)
        print("Organizing Wheat Leaf dataset...")
        print("=" * 60)
        stats_wheat = organize_wheat_dataset(wheat_source, target_raw_dir)
        print(f"\n[OK] Wheat: {stats_wheat['images_moved']} images, {len(stats_wheat['diseases'])} diseases")
    
    # Summary
    print("\n" + "=" * 60)
    print("Organization Complete!")
    print("=" * 60)
    print(f"\nOrganized dataset location: {target_raw_dir}")
    print("\nStructure:")
    print("data/raw/")
    print("├── wheat/")
    print("│   ├── healthy/")
    print("│   ├── septoria/")
    print("│   └── stripe_rust/")
    print("├── maize/")
    print("│   ├── healthy/")
    print("│   └── ...")
    print("└── ...")
    
    # Ask if user wants to create train/val/test splits
    print("\n" + "=" * 60)
    response = input("Create train/val/test splits? (y/n): ").strip().lower()
    if response == 'y':
        create_train_val_split(target_raw_dir)
        print("\n[OK] Splits created in data/raw/train/, data/raw/val/, data/raw/test/")
    else:
        print("\n[INFO] Skipping split creation. You can use the dataset directly from data/raw/")
    
    print("\n" + "=" * 60)
    print("Done! You can now use the dataset for training.")
    print("=" * 60)


if __name__ == "__main__":
    main()

