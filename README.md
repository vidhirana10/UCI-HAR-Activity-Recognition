# UCI HAR Dataset - Activity Recognition

## Overview
This repository contains a Jupyter Notebook that explores **Human Activity Recognition (HAR)** using the **UCI HAR dataset**. It compares **machine learning (ML) and deep learning (DL) approaches** to classify human activities based on motion sensor data.

## Features Implemented
- **Deep Learning:** Long Short-Term Memory (LSTM) and Convolutional Neural Networks (CNN) using PyTorch.
- **Feature Engineering:** Extracting time-series features using **TSFEL (Time Series Feature Extraction Library).**
- **Machine Learning Models:** Random Forest, Support Vector Machine (SVM), and Logistic Regression.
- **Performance Comparison:** Evaluation of ML and DL models using accuracy, precision, recall, and F1-score.
- **Visualizations:** Charts and confusion matrices to compare model performances.

## Dataset
The **UCI HAR Dataset** consists of motion sensor readings (accelerometer and gyroscope) collected from smartphones carried by subjects performing different activities (e.g., walking, sitting, lying, standing, etc.).

## Installation & Requirements
To run this notebook, install the required dependencies:
```bash
pip install -r requirements.txt
```
If using Google Colab, ensure to mount Google Drive before accessing the dataset.

## Notebook Workflow
1. **Load Dataset:** The dataset is loaded and preprocessed.
2. **Feature Extraction:** TSFEL is used to extract meaningful time-series features.
3. **Machine Learning Models:** Traditional ML models (Random Forest, SVM, Logistic Regression) are trained and evaluated.
4. **Deep Learning Models:** LSTM and CNN architectures are implemented and trained using PyTorch.
5. **Performance Comparison:** Models are compared using various evaluation metrics and visualized using confusion matrices and accuracy plots.

## Results & Conclusion
1. **Random Forest** is the best-performing model among all the tested models, with an accuracy of approximately **85%**. This indicates that Random Forest effectively captures patterns in the data for classifying human activities.
2. **Logistic Regression** demonstrates decent performance with an accuracy score of approximately **76%**, indicating its ability to model the relationship between the features and the activity classes.
3. **SVM and CNN** exhibit similar performance, both achieving approximately **61% accuracy**. This suggests that while these models are capable of classification, their performance is inferior to Random Forest and Logistic Regression in this specific case.
4. **LSTM** shows the lowest accuracy among the tested models (**around 54%**). This implies that the LSTM, in its current configuration and with the provided data, struggles to learn the temporal dependencies within the time series data as effectively as the other methods.

Overall, **Random Forest is the top performer**, suggesting its effectiveness for this classification problem. **Logistic Regression** also provides solid results. Deep learning models (**LSTM and CNN**) did not outperform the simpler methods in this context, possibly due to factors like hyperparameters, data preprocessing, or architecture choices. Further analysis and tuning of these deep learning models could potentially improve their performance.

### Visualizations
Below are key visualizations from the analysis:



#### Comparison of various ML models when trained on original raw sensor data, a pre-extracted feature set provided by the dataset authors, and a feature set generated using the TSFEL library
![image](https://github.com/user-attachments/assets/291b7d41-ae7a-458f-9748-f83c8c061986)


#### Comparison for accuracies of DL models
![image](https://github.com/user-attachments/assets/dd8ed89f-ff1d-4029-8453-f36d874f62aa)


#### Comparison for accuracies of DL versus ML models
![image](https://github.com/user-attachments/assets/02b4a384-ab89-4110-9cc5-f22a2d771b16)




## How to Use
Clone this repository and run the Jupyter Notebook in your preferred environment.
```bash
git clone https://github.com/yourusername/UCI-HAR-Activity-Recognition.git
cd UCI-HAR-Activity-Recognition
jupyter notebook Selection_task_submission.ipynb
```



## Contact
For any queries, feel free to reach out or contribute to this project!



