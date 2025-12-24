#!/usr/bin/env python3
"""
Script to download and prepare PlantVillage dataset for training.

This script helps download the PlantVillage dataset from Kaggle and
organize it into the expected directory structure.
"""

import os
import sys
from pathlib import Path
import zipfile
import shutil

def download_from_kaggle(dataset_name="abdallahalidev/plantvillage-dataset", output_dir="data/raw"):
    """
    Download PlantVillage dataset from Kaggle.
    
    Requires: pip install kaggle
    Setup: Place kaggle.json in ~/.kaggle/ (see https://www.kaggle.com/docs/api)
    """
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install with: pip install kaggle")
        print("\nAlso, you need to:")
        print("1. Create a Kaggle account")
        print("2. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("3. Place kaggle.json in ~/.kaggle/ (or C:/Users/<username>/.kaggle/ on Windows)")
        return False
    
    print(f"Downloading {dataset_name} from Kaggle...")
    print("This may take a while (dataset is ~1.5 GB)...")
    
    try:
        kaggle.api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True
        )
        print(f"✓ Dataset downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        return False


def organize_plantvillage_structure(data_dir="data/raw"):
    """
    Organize PlantVillage dataset into crop/disease structure.
    
    PlantVillage structure: color/class_name/image.jpg
    Target structure: crop/disease/image.jpg
    """
    data_path = Path(data_dir)
    
    # Find PlantVillage folders
    plantvillage_dirs = list(data_path.glob("*PlantVillage*"))
    if not plantvillage_dirs:
        plantvillage_dirs = list(data_path.glob("color"))
    
    if not plantvillage_dirs:
        print("Could not find PlantVillage dataset directory.")
        print(f"Expected structure: {data_dir}/color/ or {data_dir}/*PlantVillage*/")
        return False
    
    print("Organizing PlantVillage dataset structure...")
    
    for pv_dir in plantvillage_dirs:
        if not pv_dir.is_dir():
            continue
            
        # Process color subdirectory if it exists
        color_dir = pv_dir / "color"
        if not color_dir.exists():
            color_dir = pv_dir
        
        # Find all class directories
        class_dirs = [d for d in color_dir.iterdir() if d.is_dir()]
        
        organized_count = 0
        for class_dir in class_dirs:
            # Parse class name: e.g., "Corn_(maize)___Common_rust"
            class_name = class_dir.name
            parts = class_name.split("___")
            
            if len(parts) >= 2:
                crop = parts[0].replace("_", " ").strip().lower()
                disease = parts[1].replace("_", " ").strip().lower()
            else:
                # Fallback: use class name as crop, "healthy" as disease
                crop = class_name.replace("_", " ").strip().lower()
                disease = "healthy"
            
            # Normalize crop names
            crop_mapping = {
                "corn (maize)": "maize",
                "corn": "maize",
                "cherry (including sour)": "cherry",
                "pepper bell": "pepper",
            }
            crop = crop_mapping.get(crop, crop)
            
            # Create target directory
            target_dir = data_path / crop / disease
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_file in image_files:
                shutil.copy2(img_file, target_dir / img_file.name)
                organized_count += 1
            
            print(f"  Organized {len(image_files)} images: {crop}/{disease}")
        
        print(f"\n✓ Organized {organized_count} images total")
        return True
    
    return False


def create_annotations_csv(data_dir="data/raw", output_file="data/splits/all_annotations.csv"):
    """
    Create annotations CSV file from organized dataset structure.
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import csv
    
    annotations = []
    label_map = {}
    current_label = 0
    
    # Traverse crop/disease structure
    for crop_dir in sorted(data_path.iterdir()):
        if not crop_dir.is_dir() or crop_dir.name.startswith('.'):
            continue
        
        crop = crop_dir.name
        for disease_dir in sorted(crop_dir.iterdir()):
            if not disease_dir.is_dir():
                continue
            
            disease = disease_dir.name
            class_name = f"{crop}/{disease}"
            
            if class_name not in label_map:
                label_map[class_name] = current_label
                current_label += 1
            
            # Get all images
            image_files = list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.png"))
            
            for img_file in image_files:
                annotations.append({
                    'image_path': str(img_file.relative_to(data_path.parent)),
                    'label': label_map[class_name],
                    'crop': crop,
                    'disease': disease
                })
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'crop', 'disease'])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"✓ Created annotations file: {output_file}")
    print(f"  Total images: {len(annotations)}")
    print(f"  Total classes: {len(label_map)}")
    
    return True


def main():
    """Main function to download and prepare PlantVillage dataset."""
    print("=" * 60)
    print("PlantVillage Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check if dataset already exists
    data_dir = Path("data/raw")
    if data_dir.exists() and any(data_dir.iterdir()):
        response = input(f"Dataset directory {data_dir} already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    
    # Download
    print("\n[1/3] Downloading dataset...")
    download_success = download_from_kaggle(output_dir=str(data_dir))
    
    if not download_success:
        print("\nYou can download manually and place in data/raw/")
        print("Then run this script again to organize the structure.")
        return
    
    # Organize structure
    print("\n[2/3] Organizing dataset structure...")
    organize_plantvillage_structure(data_dir=str(data_dir))
    
    # Create annotations
    print("\n[3/3] Creating annotations CSV...")
    create_annotations_csv(data_dir=str(data_dir))
    
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the dataset: python -m notebooks.01_data_exploration")
    print("2. Split into train/val/test sets")
    print("3. Start training: python train.py --model mobilenet_v3")


if __name__ == "__main__":
    main()

