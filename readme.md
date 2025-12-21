SmartCrop AI: Intelligent Multimodal Crop Disease Detection and Growth Assistant
 Problem Statement:
 Many Ethiopian farmers lose a significant portion of their harvest due to delayed detection of crop diseases and a lack of timely agricultural guidance. Access to agricultural experts is limited, and most farmers rely on visual observation, which often identifies problems too late for effective treatment. There is also a language barrier, as most available tools are not localized for Amharic or regional dialects. 
Objective: 
This project aims to develop an intelligent, AI-powered mobile application that assists farmers in monitoring and improving crop health using voice, text, and real-time video input. The tool will provide responses in both Amharic (local language) and English, depending on the farmer’s preference. 
Key features include:
 Voice and Text Interaction: Farmers can ask questions about their crops using either voice or text in Amharic or English.
 Real-Time Video & Image Analysis: The app can detect crop diseases instantly by analyzing photos or live video from the farmer’s smartphone camera. 
Personalized Crop Guidance: Offers step-by-step recommendations for treatment, watering, fertilizer use, and harvesting.
 GPS-Based Localization: Provides region-specific agricultural advice based on the farmer’s location and local climate conditions. 
Voice and Text Responses: The system communicates results and suggestions back to the farmer using synthesized voice and text in their preferred language. 
Disease Severity Estimation: Don’t just detect disease — estimate severity using:
YOLO lesion counts, SAM segmentation (leaf area), Percentage of leaf damage
Outputs:
Mild (0–20%)
Moderate (20–50%)
Severe (50–100%)
Explainability with Heatmaps : Use Grad-CAM to highlight where the disease is.This helps:
farmer trust, expert verification and dataset improvement

Significance:
 Empowers Farmers: Enables early detection of crop diseases, helping farmers take quick action to prevent losses.
 Local Language Accessibility: Supports Amharic and English for inclusivity and ease of use.
 Improves Productivity: Provides precise and timely recommendations, increasing overall yield and crop quality. 
Smart & Location-Aware: Uses GPS to adapt advice based on soil, weather, and local agricultural conditions. 
Promotes AI Adoption in Agriculture: Demonstrates how AI, IoT, and mobile technology can work together to modernize farming practices in Ethiopia and beyond.


SMARTCROP AI: Multimodal Crop Disease Detection & Growth Assistant for Ethiopian Agriculture
Final Year Project Specification Document (≈40 pages)
Authors: Temesgen Berhanu, Lemma Getiye, Tekleeyesus Munye
Department: Software Engineering
Date: 2025

TABLE OF CONTENTS
1.	Abstract
2.	Introduction
o	2.1 Background
o	2.2 Problem Statement
o	2.3 Motivation
o	2.4 Project Objectives
o	2.5 Scope of the Project
3.	Literature Review
o	3.1 Global Trends in AI for Agriculture
o	3.2 Computer Vision in Agriculture
o	3.3 Plant Disease Detection Research
o	3.4 Growth Stage Prediction Models
o	3.5 Pretrained Agricultural Models (AgML, PlantVillage, PlantDoc, YOLO)
o	3.6 Ethiopian Agricultural Context
4.	System Requirements
o	4.1 Functional Requirements
o	4.2 Non-functional Requirements
o	4.3 Hardware/Software Requirements
5.	Methodology
o	5.1 AI Architecture Overview
o	5.2 Data Sources
o	5.3 Dataset Preparation
o	5.4 Pretrained Models Used
o	5.5 Transfer Learning & Fine-tuning Strategy
o	5.6 Model Evaluation
o	5.7 Growth Assistant Design (Rules + LLM)
6.	System Architecture
o	6.1 High-Level Architecture
o	6.2 AI Components
o	6.3 On-device vs Server Models
o	6.4 Multimodal Processing Pipeline
7.	Implementation
o	7.1 Folder Structure
o	7.2 Key Algorithms
o	7.3 Training Pipeline
o	7.4 Inference Pipeline
o	7.5 Model Export (TFLite/ONNX)
o	7.6 Mobile Integration
8.	Results & Evaluation
9.	Discussion
10.	Limitations & Challenges
11.	Conclusion
12.	Future Work
13.	References
14.	Appendices

1. ABSTRACT
Agriculture is the backbone of Ethiopia, employing more than 65% of the population. Yet, smallholder farmers suffer enormous productivity losses due to late disease identification, poor crop management practices, lack of access to agronomic knowledge, and climate variability. This project proposes SmartCrop AI, a mobile-based, multimodal artificial intelligence system that detects crop diseases using computer vision and provides a growth assistant that guides farmers through the entire life cycle of crops.
Using pretrained agricultural models (PlantVillage, PlantDoc, AgML, YOLOv8-Plant), the system performs high-accuracy disease detection with minimal computational resources. A lightweight on-device MobileNetV3 model offers offline classification, while a server-based EfficientNet-B3, YOLOv8, and AgML growth-stage classifier provide detailed, high-accuracy analysis. The assistant combines environmental data (weather, GPS), agronomic rules, and a language model to give voice/text recommendations in Amharic and English.
This document presents the complete technical plan, system architecture, implementation details, evaluation, and expected contribution.

2. INTRODUCTION
2.1 Background
Crop diseases cause up to 40–70% yield losses in Ethiopia for maize, wheat, tomatoes, teff, and other staple crops. Smallholder farmers often lack:
•	Early detection tools
•	Expert agronomy knowledge
•	Access to extension officers
•	Internet connection in remote areas
Recent advances in AI have created unprecedented opportunities in crop health monitoring, growth prediction, and farm decision support.

2.2 Problem Statement
Farmers in Ethiopia face challenges:
•	Late diagnosis of diseases
•	Lack of expert guidance
•	High cost of agronomic consultancy
•	Difficulty understanding disease severity
•	Regional language barriers
There is no low-cost, offline-ready tool that enables farmers to identify diseases and receive growth advice in their local language.

2.3 Motivation
Smart phones are increasingly available in rural Ethiopia. With pretrained agricultural AI models, even low-budget devices can run powerful vision systems. This project aims to bring that technology to Ethiopian farmers affordably.

2.4 Project Objectives
Primary Objectives
1.	Develop a smartphone-based AI system for crop disease detection using pretrained models.
2.	Build a growth assistant that provides actionable agronomy recommendations.
3.	Provide offline detection using lightweight models + online verification using stronger models.
4.	Enable multimodal interaction (image → detection, voice → questions).
Secondary Objectives
•	Provide Amharic + English support
•	Deliver explainability (Grad-CAM heatmaps)
•	Build active learning pipeline for continuous improvement

2.5 Scope
Included:
•	Plant disease detection
•	Growth stage estimation
•	Growth cycle-based recommendations
•	Cloud + offline inference
•	6–10 Ethiopian crops
•	TFLite mobile deployment
Not included:
•	Drone imagery
•	Soil sensor hardware
•	Real-time video inference

3. LITERATURE REVIEW
3.1 AI in Agriculture
Computer vision has revolutionized disease classification, pest detection, yield prediction, and plant phenotyping.
3.2 Disease Classification Research
Models like EfficientNet, MobileNet, ResNet, and YOLO are widely used.
3.3 Growth Stage Prediction
Growth stages are predicted using:
•	CNN features
•	Leaf area estimation
•	Plant height detection
•	Remote sensing features
3.4 Pretrained Agricultural Models
PlantVillage
14,000+ plant disease images.
PlantDoc
Real-field images; good for Ethiopian conditions.
AgML (Stanford)
Large universal agricultural models.
YOLOv8-Plant
Real-time disease and lesion detection.
3.6 Ethiopian Agricultural Context
Major crops:
•	Maize
•	Wheat
•	Teff
•	Sorghum
•	Tomato
•	Potato
•	Coffee
•	Barley

4. SYSTEM REQUIREMENTS
4.1 Functional Requirements
•	F1: Detect disease from image
•	F2: Mark infected areas
•	F3: Estimate growth stage
•	F4: Provide textual/voice recommendations
•	F5: Offline inference
•	F6: Online advanced inference
•	F7: Multi-language support
4.2 Non-functional
•	Fast (<500ms offline inference)
•	Lightweight (model < 10MB)
•	Secure data handling
•	85% classification accuracy

5. METHODOLOGY
5.1 AI Architecture
Two-level ensemble:
On-device model
•	MobileNetV3 pretrained on PlantVillage
•	~5MB TFLite
•	Offline detection
Server models
•	EfficientNet-B3 PlantVillage
•	YOLOv8-plant
•	AgML growth-stage classifier
•	SAM segmentation

5.2 Data Sources
•	PlantVillage
•	PlantDoc
•	AgML
•	Ethiopian field images (collected by project)

5.3 Dataset Prep
•	Clean
•	Normalize
•	Resize (224–320 px)
•	Augment (Jitter, flips, rotations)
•	Stratified train/val/test split

5.4 Pretrained Models Used
1.	EfficientNet-B3
2.	MobileNetV3
3.	ResNet50 (AgML)
4.	YOLOv8n/s
5.	SAM

5.5 Training Strategy
•	Freeze pretrained backbone
•	Train classification head
•	5–10 epochs
•	Very low GPU requirements

5.6 Growth Assistant
Uses:
•	Pretrained language model
•	Agronomy rules engine
•	Weather API
•	GPS region data

6. SYSTEM ARCHITECTURE
6.1 High-Level Diagram (text-based)
User → Mobile Camera → TFLite Model → Prediction → Growth Assistant
                ↓
       Upload to Cloud → Strong Models → Detailed Analysis

7. IMPLEMENTATION
7.1 AI Folder Structure
ai/
│
├── README.md                    # Overview of AI system, setup instructions
├── requirements.txt             # Python dependencies
├── environment.yml              # Optional (for Conda environments)
├── config/                      # Config files for training/inference
│   ├── default.yaml
│   ├── model.yaml               # Model hyperparameters
│   ├── dataset.yaml             # Paths, augmentations
│   ├── training.yaml            # LR, batch size, epochs
│   └── export.yaml              # ONNX/TFLite export settings
│
├── data/                        # NOT committed to Git (add to .gitignore)
│   ├── raw/                     # Original datasets (PlantVillage, PlantDoc, your images)
│   │   ├── maize/
│   │   ├── wheat/
│   │   ├── tomato/
│   │   └── ...
│   ├── processed/               # Preprocessed/cleaned images
│   ├── annotations/             # CVAT/LabelStudio annotations
│   ├── splits/                  # train/val/test text files
│   └── metadata/                # JSON: class-maps, label schema
│
├── notebooks/                   # Jupyter notebooks for experiments
│   ├── 01_explore_dataset.ipynb
│   ├── 02_train_classifier.ipynb
│   ├── 03_finetune_pretrained.ipynb
│   ├── 04_detect_disease_yolo.ipynb
│   ├── 05_growth_stage_model.ipynb
│   ├── 06_export_tflite.ipynb
│   └── 07_export_onnx.ipynb
│
├── src/                         # MAIN AI CODEBASE (modular & clean)
│   ├── __init__.py
│   │
│   ├── data/                    # Data loaders & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   ├── transforms.py        # Augmentations
│   │   └── preprocess.py        # Resize, normalize, cleaning
│   │
│   ├── models/                  # All model definitions & loaders
│   │   ├── __init__.py
│   │   ├── classifier_effnet.py # EfficientNet (PlantVillage pretrained)
│   │   ├── classifier_resnet.py # ResNet (AgML)
│   │   ├── yolo_detector.py     # YOLOv8 plant model wrapper
│   │   ├── segment_sam.py       # SAM for segmentation
│   │   ├── growth_stage_model.py# Crop-growth-stage classifier
│   │   └── model_utils.py       # Weights loading, freezing, etc.
│   │
│   ├── training/                # Training & fine-tuning scripts
│   │   ├── __init__.py
│   │   ├── train_classifier.py
│   │   ├── train_detector.py
│   │   ├── train_growth_stage.py
│   │   ├── loss_functions.py
│   │   ├── metrics.py
│   │   └── scheduler.py
│   │
│   ├── inference/               # Inference pipeline
│   │   ├── __init__.py
│   │   ├── predict_image.py     # Single image inference
│   │   ├── batch_inference.py   # Folder inference
│   │   ├── heatmap_cam.py       # Grad-CAM explainability
│   │   └── postprocess.py       # Confidence, severity estimation
│   │
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py            # MLflow / TensorBoard logger
│   │   ├── file_utils.py
│   │   ├── visualization.py     # Plots, confusion matrix
│   │   ├── seed.py              # Random seeds control
│   │   └── helpers.py
│   │
│   └── export/                  # Export models to mobile/server
│       ├── __init__.py
│       ├── export_tflite.py
│       ├── export_onnx.py
│       ├── optimize_quantize.py
│       └── convert_to_mobile.py
│
├── experiments/                 # Auto-logged results
│   ├── exp_001_effnet/
│   │   ├── logs/
│   │   ├── metrics.json
│   │   └── confusion_matrix.png
│   ├── exp_002_yolo/
│   └── ...
│
├── outputs/
│   ├── models/                  # Final models (.pth, .onnx, .tflite)
│   │   ├── classifier/
│   │   ├── detector/
│   │   ├── segmentation/
│   │   └── growth_stage/
│   ├── heatmaps/                # Grad-CAM results
│   └── predictions/             # Inference results (json/images)
│
├── tests/                       # Unit tests (optional but good)
│   ├── test_dataset.py
│   ├── test_model_loading.py
│   ├── test_inference.py
│   └── test_export.py
│
└── .gitignore                   # Ignore raw data, logs, model weights
7.2 Key Algorithms
•	Transfer learning
•	Multi-class classification
•	Object detection (YOLO)
•	Vision transformer embeddings
•	Growth stage classifier
•	Rule-based reasoning

7.3 Training Pipeline
•	Load pretrained model
•	Freeze layers
•	Train top layers
•	Evaluate
•	Save best model

7.4 Inference Pipeline
1.	Preprocess image
2.	Run through on-device model
3.	If confidence < threshold → send to cloud
4.	Create disease severity score
5.	Generate recommendations

7.5 Model Export
•	Convert PyTorch → ONNX
•	Convert ONNX → TFLite
•	Apply quantization

7.6 Mobile Integration
•	Flutter/React Native TFLite plugin
•	On-device prediction view
•	Image heatmaps

8. RESULTS & EVALUATION
Expected:
Metric	Target
Accuracy	85–95%
Precision	>85%
Recall	>85%
Latency	<500ms offline
Model Size	<10MB
Confusion matrix & mAP to be included after training.

9. DISCUSSION
•	High performance even on low data using pretrained models
•	On-device inference solves connectivity issues
•	Real-field conditions tested with PlantDoc + custom Ethiopian data

10. LIMITATIONS
•	Teff & sorghum datasets are limited
•	Complex diseases may require expert labeling
•	STT for Amharic can be challenging offline

11. CONCLUSION
SmartCrop AI demonstrates that pretrained models can bring advanced agricultural intelligence to Ethiopian farmers with minimal hardware. The system is accurate, efficient, and practical for rural deployment.

