#  Titanic Survival Prediction

A Machine Learning project that predicts passenger survival using the Titanic dataset.

## 📊 Project Overview
Performed Exploratory Data Analysis (EDA), data preprocessing, and feature engineering to identify patterns associated with passenger survival.

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Pickle

## 🤖 Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier

## 📈 Results

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 83.24% |
| Random Forest | 77.09% |
| Decision Tree | 73.74% |

Logistic Regression achieved the highest validation accuracy among the three models tested.

## 💾 Model Deployment
The trained model is exported using Pickle for integration into a Streamlit web application.

## 📂 Project Structure

```text
titanic-survival-prediction/
├── Titanic_dataset_analysis.ipynb
├── titanic_model.pkl
├── app.py
└── README.md
```

## 📁 Dataset
[Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

## 🚀 Future Improvements
- Hyperparameter tuning
- Improved feature engineering
