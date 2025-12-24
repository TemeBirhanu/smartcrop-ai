# SmartCrop AI - Machine Learning Module

This directory contains the AI/ML components for SmartCrop AI, including model training, inference, and export pipelines.

## 📁 Project Structure

```
ai/
├── config/              # Configuration files (YAML)
├── data/                # Datasets (not committed to Git)
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                  # Main source code
│   ├── data/            # Data loaders and preprocessing
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── inference/       # Inference pipeline
│   ├── utils/           # Utility functions
│   └── export/          # Model export (TFLite/ONNX)
├── experiments/         # Experiment results
├── outputs/             # Trained models and outputs
└── tests/               # Unit tests
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download Pretrained Models

- **SAM (Segment Anything Model)**: Download from [Meta AI](https://github.com/facebookresearch/segment-anything#model-checkpoints)
- **YOLOv8**: Automatically downloaded by ultralytics
- **EfficientNet/MobileNet**: Automatically downloaded by torchvision

### 3. Prepare Data

**📖 See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed information on where to get datasets for Ethiopian crops.**

Place your datasets in `data/raw/` following this structure:
```
data/raw/
├── maize/
│   ├── healthy/
│   ├── leaf_blight/
│   └── ...
├── wheat/
└── ...
```

**Quick start with PlantVillage dataset:**
```bash
# Download and organize PlantVillage dataset
python scripts/download_plantvillage.py
```

This will download the PlantVillage dataset (includes maize, wheat, sorghum, barley) and organize it for training.

### 4. Run Training

```bash
# Train classifier
python src/training/train_classifier.py --config config/training.yaml

# Train detector
python src/training/train_detector.py --config config/training.yaml
```

### 5. Run Inference

```bash
# Single image prediction
python src/inference/predict_image.py --image path/to/image.jpg

# Batch inference
python src/inference/batch_inference.py --folder path/to/images/
```

## 🔧 Configuration

All configuration files are in `config/`:
- `default.yaml`: General settings
- `model.yaml`: Model architectures and hyperparameters
- `dataset.yaml`: Data paths and preprocessing
- `training.yaml`: Training parameters
- `export.yaml`: Model export settings

## 📊 Models

### Pretraining Strategy

**Important**: The models use a **3-stage transfer learning** approach:

1. **ImageNet Pretrained** (already done by torchvision)
   - MobileNetV3, EfficientNet-B3, ResNet50
   - General image recognition features
   - Automatically downloaded when you create the model

2. **PlantVillage Fine-tuning** (you do this)
   - Fine-tune on PlantVillage dataset (~54K images)
   - Learn plant disease-specific patterns
   - Freeze backbone, train only classification head

3. **Ethiopian Data Fine-tuning** (you do this as you collect data)
   - Further fine-tune on Ethiopian crop images
   - Adapt to local conditions and crops

### On-Device Model
- **MobileNetV3**: Lightweight model for offline inference (~5MB TFLite)
- Starts with ImageNet pretrained → Fine-tuned on PlantVillage → Fine-tuned on Ethiopian data

### Server Models
- **EfficientNet-B3**: High-accuracy classification (ImageNet → PlantVillage → Ethiopian)
- **YOLOv8**: Disease lesion detection (COCO pretrained → Fine-tuned on lesion data)
- **SAM**: Precise segmentation (pretrained by Meta AI)
- **ResNet50**: Growth stage classification (ImageNet → Fine-tuned on growth stage data)

## 🧪 Experiments

Jupyter notebooks in `notebooks/`:
1. `01_explore_dataset.ipynb`: Dataset exploration
2. `02_train_classifier.ipynb`: Classification training
3. `03_finetune_pretrained.ipynb`: Transfer learning
4. `04_detect_disease_yolo.ipynb`: YOLO training
5. `05_growth_stage_model.ipynb`: Growth stage classifier
6. `06_export_tflite.ipynb`: TFLite export
7. `07_export_onnx.ipynb`: ONNX export

## 📝 Notes

- Large datasets and model weights are excluded via `.gitignore`
- Use `data/.gitkeep` to preserve directory structure
- Experiment results are logged to `experiments/`
- Final models are saved to `outputs/models/`



