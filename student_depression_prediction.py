# ==========================================
# Import Libraries
# ==========================================

# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Student Depression Dataset.csv")

# ==========================================
# Data Understanding
# ==========================================

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# Data Cleaning
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# ==========================================
# Target Distribution
# ==========================================

print("\nDepression Distribution:")
print(df["Depression"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Depression", data=df)
plt.title("Distribution of Depression")
plt.show()

# ==========================================
# EDA
# ==========================================

# Academic Pressure vs Depression
plt.figure(figsize=(6,4))
sns.violinplot(
    x="Depression",
    y="Academic Pressure",
    data=df,
    palette="Set2"
)
plt.title("Academic Pressure vs Depression")
plt.show()

# Study Satisfaction vs Depression
plt.figure(figsize=(8,6))
sns.boxplot(
    x="Depression",
    y="Study Satisfaction",
    data=df
)
plt.title("Study Satisfaction vs Depression")
plt.show()

# Work/Study Hours vs Depression
plt.figure(figsize=(8,6))
sns.boxplot(
    x="Depression",
    y="Work/Study Hours",
    data=df
)
plt.title("Work/Study Hours vs Depression")
plt.show()



# ==========================================
# Correlation Heatmap
# ==========================================

corr = df.corr(numeric_only=True)

plt.figure(figsize=(24,10))
sns.heatmap(corr,annot=True,fmt=".2f",cmap="Blues",linewidths=0.5)

plt.title("Correlation Matrix")
plt.show()

# ==========================================
# Encoding Categorical Features
# ==========================================

encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# ==========================================
# Features and Target
# ==========================================

if "id" in df.columns:
    X = df.drop(columns=["Depression", "id"])
else:
    X = df.drop(columns=["Depression"])

print(X.columns)

y = df["Depression"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Logistic Regression
# ==========================================

lr_model = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n===== Logistic Regression =====")
print("Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_pred))

lr_cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6,5))
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression")
plt.show()


# ==========================================
# Random Forest
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n===== Random Forest =====")
print("Accuracy:", rf_accuracy)
print(classification_report(y_test, rf_pred))

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest")
plt.show()


# ==========================================
# SVM
# ==========================================

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\n===== SVM =====")
print("Accuracy:", svm_accuracy)
print(classification_report(y_test, svm_pred))

svm_cm = confusion_matrix(y_test, svm_pred)

print("\nConfusion Matrix:")
print(svm_cm)

plt.figure(figsize=(6,5))
sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Oranges")
plt.title("SVM")
plt.show()

import joblib
# ==========================================
# Save Best Model
# ==========================================

best_model = lr_model
best_accuracy = lr_accuracy

if rf_accuracy > best_accuracy:
    best_accuracy = rf_accuracy
    best_model = rf_model

if svm_accuracy > best_accuracy:
    best_accuracy = svm_accuracy
    best_model = svm_model

print("\n==============================")
print("Model Comparison")
print("==============================")
print("Logistic Regression :", lr_accuracy)
print("Random Forest       :", rf_accuracy)
print("SVM                 :", svm_accuracy)

print("\nBest Accuracy :", best_accuracy)
print("Best Model    :", type(best_model).__name__)

joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoder.pkl")

print("Best model saved successfully!")
