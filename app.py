
# =========================================
# Import Libraries
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load Dataset

df = pd.read_csv("StudentPerformanceFactors.csv")

print(df.head())
print(df.shape)
print(df.info())
#churn
df["Churn"] = df["Exam_Score"].apply(lambda x: 1 if x < 60 else 0)

print(df["Churn"].value_counts())

# Data Cleaning checking null values and filling using mode

print(df.isnull().sum())

df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0], inplace=True)
df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0], inplace=True)
df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0], inplace=True)

print(df.isnull().sum())

# Feature Engineering is the process of creating the new features from the existing ones.

# how efficient student is in studies
df["Study_Efficiency"] = df["Previous_Scores"] / (df["Hours_Studied"] + 1)

# overall engagement in academics
df["Academic_Engagement"] = df["Attendance"] + df["Hours_Studied"]

# physical + mental balance
df["Wellness_Score"] = df["Sleep_Hours"] + df["Physical_Activity"]

# risk score (low attendance + low marks = high risk)
df["Risk_Score"] = (100 - df["Attendance"]) + (100 - df["Previous_Scores"])

# study vs sleep balance
df["Study_Sleep_Ratio"] = df["Hours_Studied"] / (df["Sleep_Hours"] + 1)

print(df[["Study_Efficiency", "Academic_Engagement", "Wellness_Score", "Risk_Score", "Study_Sleep_Ratio"]].head())

# EDA

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

sns.histplot(df["Exam_Score"], bins=20, kde=True)
plt.title("Exam Score Distribution")
plt.show()

sns.histplot(df["Attendance"], bins=20, kde=True)
plt.title("Attendance Distribution")
plt.show()

sns.histplot(df["Hours_Studied"], bins=20, kde=True)
plt.title("Hours Studied Distribution")
plt.show()

sns.countplot(x="Gender", hue="Churn", data=df)
plt.title("Churn by Gender")
plt.show()

sns.countplot(x="Motivation_Level", hue="Churn", data=df)
plt.title("Churn by Motivation Level")
plt.show()

sns.countplot(x="School_Type", hue="Churn", data=df)
plt.title("Churn by School Type")
plt.show()

sns.boxplot(x="Churn", y="Exam_Score", data=df)
plt.title("Exam Score vs Churn")
plt.show()

#heatmap

df_heatmap = df.copy()
le = LabelEncoder()

for col in df_heatmap.select_dtypes(include="object").columns:
    df_heatmap[col] = le.fit_transform(df_heatmap[col])

plt.figure(figsize=(12,8))
sns.heatmap(df_heatmap.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Preprocessing (encoding categorical data)

df_model = df.copy()

for col in df_model.select_dtypes(include="object").columns:
    df_model[col] = le.fit_transform(df_model[col])
# Feature Selection
# removing Exam_Score to avoid data leakage ❗
# in dataset the output column that should contain churn was no their so we have intially created a chun on the basis of exam_score,
# otherwise the model will know the churn value
X = df_model.drop(["Churn", "Exam_Score"], axis=1)
y = df_model["Churn"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
# used for binary classification
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Decision Tree
# rule-based model

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Compare both models

lr_acc = accuracy_score(y_test, y_pred_lr)
dt_acc = accuracy_score(y_test, y_pred_dt)

print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)

if lr_acc > dt_acc:
    print("Best Model: Logistic Regression")
else:
    print("Best Model: Decision Tree")

# accuracy comparison graph

models = ["Logistic Regression", "Decision Tree"]
accuracies = [lr_acc, dt_acc]

plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
# roc curve
y_prob = lr.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = " + str(round(roc_auc,2)))
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nInsights:")
print("Students with low attendance and low previous scores are more likely to dropout.")
print("Higher study hours and good engagement reduce dropout chances.")
print("Balanced lifestyle (sleep + activity) helps in better performance.")
