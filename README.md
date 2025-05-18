# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to detect fraudulent credit card transactions using machine learning techniques. It addresses real-world challenges like class imbalance and real-time prediction using a user-friendly web interface.

---

## 📌 Project Overview

- **Objective**: Build a reliable model to detect credit card fraud with high accuracy.
- **Approach**: Classification with imbalanced data handling, model evaluation, and a deployed UI.
- **Tech Stack**: Python, scikit-learn, imbalanced-learn, Streamlit, Pandas, Matplotlib, Seaborn.

---

## 🚀 Features

- Handles highly imbalanced dataset using **SMOTE**.
- Trains a **Random Forest Classifier** for prediction.
- Evaluates performance with **precision, recall, F1-score, and ROC-AUC**.
- Deploys a real-time prediction interface using **Streamlit**.
- Modular project structure with reusable components.

---

## 📂 Project Structure

credit-card-fraud-detection/
│
├── data/
│ └── creditcard.csv # Input dataset
│
├── src/
│ ├── data_preprocessing.py # Data loading & preprocessing
│ ├── train_model.py # Model training & saving
│ ├── evaluate_model.py # Evaluation metrics and visuals
│ └── predict.py # Prediction function for UI
│
├── run_train.py # Script to preprocess, train, evaluate
├── model.pkl # Saved trained model
├── app.py # Streamlit UI
├── requirements.txt # Python dependencies
└── README.md # Project overview

---

## 📊 Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Contains anonymized features from PCA (V1 to V28), `Time`, `Amount`, and target variable `Class`.
- `Class` = 1 → Fraud, `Class` = 0 → Legit.

---

## 📈 Model Performance

- **Classifier**: Random Forest
- **Recall (Fraud Class)**: ~84%
- **Precision (Fraud Class)**: ~85%
- **ROC-AUC Score**: ~0.97
- **Accuracy**: 100% (caution: class imbalance)

---

## 🧪 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

### 2. Install requirements 

pip install -r requirements.txt

### 3. Train the model

python run_train.py

### 4. Launch the UI

streamlit run app.py

📃 License

This project is for academic use under the Sarala Birla University B.Tech Minor Project 2025 guidelines.
