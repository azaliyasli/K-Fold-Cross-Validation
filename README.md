# K-Fold Cross-Validation for SVM Classifier

## 📌 Overview
This project applies **Support Vector Machine (SVM)** classification with an **RBF kernel** to predict whether a user purchases a product based on their **age** and **estimated salary**. The **main focus** of this project is on **K-Fold Cross Validation**, which is used to measure the **robustness** and **stability** of the trained model across multiple splits of the training data.

## ✨ Features
- Load and preprocess dataset
- Feature scaling using `StandardScaler`
- Train an SVM classifier with RBF kernel
- Make predictions on test data
- Evaluate performance with:
  - **K-Fold Cross Validation (primary evaluation method)**
  - Accuracy on the test set

## 📂 Dataset
The `Social_Network_Ads.csv` file contains:
- **Age**
- **Estimated Salary**
- **Purchased** (target: 0 = No, 1 = Yes)

## 🛠 Requirements
- Python 3.x
- pandas
- scikit-learn
