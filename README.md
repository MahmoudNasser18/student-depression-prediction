# 🧠 Student Depression Prediction System

A Machine Learning web application built with **Streamlit** to predict whether a student is likely to experience depression based on academic, personal, and lifestyle factors.

---

## 📌 Project Overview

This project uses a trained **Support Vector Machine (SVM)** model to predict student depression.

The application allows users to enter student information and instantly receive a prediction along with the model confidence.

---

## 🚀 Features

- Modern Streamlit interface
- Predicts Depression / No Depression
- Displays prediction confidence
- Uses trained Machine Learning model
- Automatic categorical encoding
- Automatic feature scaling
- Easy to use

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 📊 Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Label Encoding
4. Feature Scaling
5. Train/Test Split
6. Model Training
7. Model Evaluation
8. Model Deployment using Streamlit

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

The **Support Vector Machine (SVM)** achieved the best performance and was selected for deployment.

---

## 📁 Project Structure

```
Student_Depression_Project/
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── Student_Depression.ipynb
```

---

## ▶️ Installation

Clone the repository

```bash
git clone <repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application

The application allows users to enter:

- Gender
- Age
- City
- Profession
- Academic Pressure
- Work Pressure
- CGPA
- Study Satisfaction
- Job Satisfaction
- Sleep Duration
- Dietary Habits
- Degree
- Suicidal Thoughts
- Work/Study Hours
- Financial Stress
- Family History of Mental Illness

Then predicts whether the student is likely to experience depression.

---

## 📈 Model Output

The application displays:

- Prediction Result
- Model Confidence
- Input Summary

---

## ⚠️ Disclaimer

This application is intended for educational purposes only.

It does **not** provide medical diagnosis or replace professional mental health assessment.

---

## 👨‍💻 Developed By

Mahmoud