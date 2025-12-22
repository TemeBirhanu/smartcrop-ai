# Jupyter Notebooks

This directory contains example notebooks for experimenting with SmartCrop AI.

## Notebooks

### 1. `01_data_exploration.ipynb`
- Explore and visualize your dataset
- Check class distribution
- Verify image sizes and quality
- Display sample images from each class

### 2. `02_train_model.ipynb`
- Train a crop disease classifier
- Use pretrained models (MobileNetV3 or EfficientNet-B3)
- Fine-tune with frozen backbone (minimal data needed)
- Visualize training curves

### 3. `03_inference_example.ipynb`
- Run inference on test images
- Generate Grad-CAM heatmaps for explainability
- Batch inference on multiple images
- Visualize predictions

### 4. `04_model_evaluation.ipynb`
- Evaluate trained models with detailed metrics
- Generate confusion matrix
- Calculate per-class precision, recall, F1-score
- Classification report

## Usage

1. Install Jupyter:
```bash
pip install jupyter notebook
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open and run notebooks in order (01 → 02 → 03 → 04)

## Notes

- Update paths in each notebook to match your dataset structure
- Adjust `NUM_CLASSES` and `CLASS_NAMES` based on your dataset
- Notebooks assume data is organized in `../data/raw/` directory


