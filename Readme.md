```markdown
# TSCA-ViT for Medical Image Segmentation

This repository implements **TSCA-ViT**, a novel transformer-based architecture for medical image segmentation, featuring dual attention mechanisms and skip connections for enhanced spatial awareness and segmentation accuracy. It includes training, testing, preprocessing pipelines, and evaluation metrics tailored for medical imaging datasets.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Results and Metrics](#results-and-metrics)
- [File Structure](#file-structure)
- [Acknowledgments](#acknowledgments)

---

## Features

1. **TSCA-ViT Architecture**: 
   - Dual attention mechanisms (channel and spatial).
   - Cross-attention skip connections for precise spatial information retention.
2. **Baseline Comparison**:
   - Implements UNet for benchmark evaluation.
3. **Comprehensive Preprocessing**:
   - Includes flipping, rotation, Gaussian noise, blurring, and dataset-wide normalization.
4. **Robust Metrics**:
   - Dice Similarity Coefficient (DSC), Intersection over Union (IoU), Accuracy, Precision, and Recall.

---

## Requirements

To set up the environment, install the following dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- Python 3.8+
- PyTorch 1.10+
- TensorBoard
- NumPy
- imgaug
- tqdm
- matplotlib
- pandas

---

## Usage

### Training

Run the following command to train the model:

```bash
python train.py --root_path <path_to_train_data> --list_dir <path_to_list_dir> --output_dir <path_to_output>
```

### Training Arguments

| Argument           | Default Value                        | Description                                        |
|---------------------|--------------------------------------|--------------------------------------------------|
| `--root_path`       | `data/Synapse/train_npz`            | Path to training data.                           |
| `--list_dir`        | `./lists/lists_Synapse`             | Path to dataset splits list.                     |
| `--num_classes`     | `9`                                 | Number of segmentation classes.                  |
| `--batch_size`      | `24`                                | Batch size.                                      |
| `--img_size`        | `224`                               | Input image size.                                |
| `--max_epochs`      | `400`                               | Number of training epochs.                       |
| `--base_lr`         | `0.05`                              | Base learning rate.                              |
| `--output_dir`      | `./model_out`                       | Directory for saving models and logs.            |

---

### Testing

To evaluate the model on test data:

```bash
python test.py --volume_path <path_to_test_data> --output_dir <path_to_model> --test_save_dir <path_to_predictions>
```

### Testing Arguments

| Argument           | Default Value                        | Description                                        |
|---------------------|--------------------------------------|--------------------------------------------------|
| `--volume_path`     | `/images/PublicDataset/.../data`     | Path to the test data.                           |
| `--num_classes`     | `9`                                 | Number of segmentation classes.                  |
| `--output_dir`      | `./model_out`                       | Path to the saved model checkpoints.             |
| `--test_save_dir`   | `../predictions`                    | Directory for saving test results.               |

---

## Results and Metrics

During evaluation, the following metrics are computed and logged:

1. **Accuracy**: The ratio of correctly classified pixels to the total number of pixels.
2. **Precision**: The ratio of true positives to the sum of true positives and false positives.
3. **Recall**: The ratio of true positives to the sum of true positives and false negatives.
4. **Dice Similarity Coefficient (DSC)**: Measures overlap between predicted and true masks.
5. **Intersection over Union (IoU)**: Ratio of intersection area to the union area of predicted and true masks.

---

## File Structure

```plaintext
project/
├── datasets/
│   └── dataset_osteosarcoma.py   # Dataset and augmentation techniques
├── networks/
│   └── TSCAFormer.py             # TSCA-ViT architecture
├── trainer.py                    # Training and evaluation functions
├── train.py                      # Training script
├── test.py                       # Testing script
├── utils.py                      # Utility functions (e.g., metric computation)
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation
```

---

## Acknowledgments

This project is inspired by the Vision Transformer (ViT) architecture and state-of-the-art research in medical image segmentation. Special thanks to contributions from open-source repositories and frameworks that facilitated the development of TSCA-ViT.
```