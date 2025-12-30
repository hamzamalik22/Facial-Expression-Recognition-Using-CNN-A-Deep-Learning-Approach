# Facial Expression Recognition using CNN: A Deep Learning Approach

This repository contains the implementation of a research study evaluating three Convolutional Neural Network (CNN) architectures for automatic emotion classification. The project demonstrates the progression from a custom baseline CNN to advanced transfer learning models like VGG19 and ResNet18.

## 👥 Authors
* **Hamza Abdul Jabbar** (22-CS-086)
* **Muhammad Hassan Azmat** (22-CS-15)
* **Hasnain Ali** (22-CS-143)

## 📌 Project Overview
Facial Expression Recognition (FER) is essential for human-computer interaction, healthcare, and surveillance. This research utilizes the **FER-2013 dataset**, consisting of 48x48 pixel grayscale images categorized into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

### Key Methodology
**Baseline CNN:** A custom architecture built with TensorFlow/Keras.
**Transfer Learning:** VGG19 and ResNet18 architectures implemented in PyTorch, pre-trained on ImageNet.
**Advanced Augmentation:** Implementation of `RandomCrop(44)` for training and `TenCrop(44)` for robust testing.
**Regularization:** Use of Batch Normalization, Dropout (0.5), and Early Stopping to mitigate overfitting.



## 📊 Performance Results
The research confirmed that deeper architectures and transfer learning significantly improve classification accuracy for subtle expressions.

| Model | Framework | Test Accuracy | Strengths |
| :--- | :--- | :--- | :--- |
| **Baseline CNN** | TensorFlow | 67.25% | Lightweight & efficient |
| **VGG19** | PyTorch | 70.31% | Deep features, strong on dominant classes |
| **ResNet18** | PyTorch | **71.60%** | Best Performance, superior generalization |



## 🛠️ Installation & Usage
### 1. Prerequisites
Install the required libraries:
```bash
pip install -r requirements.txt

```

### 2. Dataset

Download the **FER-2013** dataset from Kaggle and place the images in the `data/` directory.

### 3. Running the Project

* **Interactive:** Open `notebooks/Facial_Expression_Recognition_CNN.ipynb` to see the step-by-step implementation.
* **Modular:** Execute the PyTorch training pipeline:

```bash
python src/train_pytorch.py

```

## 📄 Documentation

For a detailed analysis of the results, confusion matrices, and confidence audits, please refer to the **[final_report.pdf](Final_Report.pdf)** included in this repository.

## 📜 License

This project is licensed under the MIT License.

