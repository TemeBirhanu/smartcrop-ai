"""
Check what classes your trained model knows
"""

import torch
from pathlib import Path
import json

def check_model_classes(checkpoint_path: str):
    """Check classes in trained model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("=" * 60)
    print("MODEL CLASS INFORMATION")
    print("=" * 60)
    
    # Get number of classes
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    num_classes = None
    
    for key in reversed(list(state_dict.keys())):
        if 'classifier' in key and 'weight' in key:
            num_classes = state_dict[key].shape[0]
            print(f"\n✓ Number of classes: {num_classes}")
            break
    
    # Check if class names are saved
    if 'class_names' in checkpoint:
        print("\n✓ Class names found in checkpoint:")
        for i, name in enumerate(checkpoint['class_names']):
            print(f"  {i:2d}: {name}")
    else:
        print("\n⚠️  Class names not saved in checkpoint")
        print("   Classes are indexed 0 to", num_classes - 1)
        print("\n   To see actual class names, check your training dataset structure:")
        print("   python scripts/check_dataset_classes.py")
    
    # Check training info
    if 'best_val_acc' in checkpoint:
        print(f"\n✓ Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"✓ Trained for {checkpoint['epoch']} epochs")
    
    print("\n" + "=" * 60)

def check_dataset_classes(data_dir: str = "data/raw/train"):
    """Check classes in dataset structure."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️  Dataset directory not found: {data_dir}")
        return
    
    print("=" * 60)
    print("DATASET CLASS STRUCTURE")
    print("=" * 60)
    
    crops = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    all_classes = []
    for crop in crops:
        crop_path = data_path / crop
        diseases = sorted([d.name for d in crop_path.iterdir() if d.is_dir()])
        
        print(f"\n{crop.upper()}:")
        for disease in diseases:
            class_name = f"{crop}_{disease}"
            all_classes.append(class_name)
            print(f"  - {class_name}")
    
    print(f"\n✓ Total classes: {len(all_classes)}")
    print(f"✓ Total crops: {len(crops)}")
    
    # Save class map
    class_map = {name: idx for idx, name in enumerate(sorted(all_classes))}
    output_file = Path("data/metadata/class_map.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(class_map, f, indent=2)
    
    print(f"\n✓ Saved class map to: {output_file}")
    print("=" * 60)
    
    return all_classes

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "outputs/models/checkpoints/best_model.pth"
    
    if Path(checkpoint_path).exists():
        check_model_classes(checkpoint_path)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    print("\n")
    check_dataset_classes()

