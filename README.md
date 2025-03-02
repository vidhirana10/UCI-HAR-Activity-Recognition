# UCI HAR Dataset - Activity Recognition

## Overview
This repository contains a Jupyter Notebook that explores **Human Activity Recognition (HAR)** using the **UCI HAR dataset**. It compares **machine learning (ML) and deep learning (DL) approaches** to classify human activities based on motion sensor data.

## Features Implemented
- **Deep Learning:** Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN) using PyTorch.
- **Feature Engineering:** Extracting time-series features using **TSFEL (Time Series Feature Extraction Library).**
- **Machine Learning Models:** Random Forest, Support Vector Machine (SVM), and Logistic Regression.
- **Performance Comparison:** Evaluation of ML and DL models using accuracy, precision, recall, and F1-score.

## Dataset
The **UCI HAR Dataset** consists of motion sensor readings (accelerometer and gyroscope) collected from smartphones carried by subjects performing different activities (e.g., walking, sitting, lying, standing, etc.).

## Installation & Requirements
To run this notebook, install the required dependencies:
```bash
pip install torch torchvision torchaudio
pip install scikit-learn tsfel
```
If using Google Colab, ensure to mount Google Drive before accessing the dataset.

## Notebook Workflow
1. **Load Dataset:** The dataset is loaded and preprocessed.
2. **Feature Extraction:** TSFEL is used to extract meaningful time-series features.
3. **Machine Learning Models:** Traditional ML models (Random Forest, SVM, Logistic Regression) are trained and evaluated.
4. **Deep Learning Models:** LSTM and CNN architectures are implemented and trained using PyTorch.
5. **Performance Comparison:** Models are compared using various evaluation metrics and visualized using confusion matrices.

## Results & Conclusion
- **Feature engineering** improved ML models' accuracy.
- **LSTM and CNN outperformed ML models**, demonstrating deep learning's ability to capture complex temporal relationships.
- **CNN had a slight edge over LSTM**, but results may vary based on hyperparameters and dataset size.
- **Future Work:** Optimizing architectures and experimenting with advanced feature extraction techniques could further improve performance.

## How to Use
Clone this repository and run the Jupyter Notebook in your preferred environment.
```bash
git clone https://github.com/vidhirana10/UCI-HAR-Activity-Recognition.git
cd UCI-HAR-Activity-Recognition
jupyter notebook Selection_task_1.ipynb
```



## Contact
For any queries, feel free to reach out or contribute to this project!

---
Feel free to modify and update this README as needed!

