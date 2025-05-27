# 💳 Task 3: Credit Card Fraud Detection using Random Forest

This project builds a machine learning pipeline to detect fraudulent credit card transactions using the **Random Forest** classifier. The dataset is highly imbalanced, so techniques like **SMOTE** are used for better performance.

---

## 📁 Dataset

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Cases**: ~0.17% (highly imbalanced)
- **Features**:
  - `Time`, `Amount`: Real-world features
  - `V1` to `V28`: PCA-anonymized features
  - `Class`: 1 = Fraud, 0 = Non-fraud

---

## 🧠 Objective

To accurately detect fraudulent transactions by:
- Handling imbalanced data
- Training a robust model
- Evaluating performance with suitable metrics

---

## 🛠️ Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **Matplotlib / Seaborn** (for visualizations)

---

## ⚙️ Workflow

1. **Load data & explore imbalance**
2. **Preprocessing**:
   - Normalize `Amount`
   - Drop unnecessary columns like `Time` (optional)
3. **Train-test split**
4. **Apply SMOTE** to balance the dataset
5. **Model training** using **Random Forest**
6. **Evaluation**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
7. **Interpretation** of false positives/negatives

---

## 🧪 Evaluation Metrics

Why not just accuracy?
- **Recall** is crucial (detect actual frauds)
- **Precision** reduces false alarms
- **F1-score** balances both

---

## 📊 Results (Example)

```plaintext
Accuracy:  0.99
Precision: 0.92
Recall:    0.89
F1-score:  0.90
