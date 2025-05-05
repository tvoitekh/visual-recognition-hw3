# NYCU Computer Vision 2025 Spring HW3
StudentID: 111550203  
Name: 提姆西

## Introduction
This project implements an instance segmentation system for cell nuclei using the Mask R-CNN architecture with MMDetection 3.x. The system is designed to:
1. Detect individual nuclei instances in microscopy images
2. Generate precise pixel-level segmentation masks for each nucleus
3. Classify the nuclei into different cell types

The approach utilizes a lightweight ResNet-18 backbone with Feature Pyramid Network (FPN) for efficient feature extraction, making it suitable for deployment in resource-constrained environments while maintaining good performance on the segmentation task.

## How to install
1. Clone this repository:
```bash
git clone https://github.com/tvoitekh/visual-recognition-hw3.git
cd visual-recognition-hw3
```

2. Make sure you have the data directory structure:
```
./hw3-data-release
├── train/
├── test_release/
└── test_image_name_to_ids.json
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

4. Run the model:
   - For training and inference:
     ```python
     python nuclei_segmentation.py
     ```
   - For inference only:
     ```python
     python nuclei_segmentation.py --skip_train
     ```
   - For visualization:
     ```python
     python nuclei_segmentation.py --visualize
     ```

## Performance snapshot
<img width="903" alt="image" src="https://github.com/user-attachments/assets/d2dd347c-27a9-4bef-b38e-c6b9cc1b2c75" />



Key performance features:
- Lightweight ResNet-18 backbone with FPN for efficient feature extraction
- Optimized anchor sizes and aspect ratios for nuclei detection
- Reduced parameters in mask head for faster inference
- Deterministic training with fixed random seed for reproducibility
- Adaptive learning rate scheduling with MultiStepLR

## Code Linting
The following commands have been run as well as manual modifications performed:
```bash
autopep8 --in-place --aggressive --max-line-length 79 nuclei_segmentation.py
```
```bash
black --line-length 79 nuclei_segmentation.py
```
<img width="756" alt="image" src="https://github.com/user-attachments/assets/cf11eeb1-b682-400a-a0e2-a94d8d9a91b2" />


As can be seen no warnings or errors are present. This verifies that the code had been successfully linted as required.
