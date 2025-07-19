import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np
import os

# Load data
df = pd.read_csv("results.csv")  # replace with the actual filename if needed

# 1. Basic Overview
print("Dataset Summary:")
print(df.describe(include='all'))
print("\nClass Distribution (Ground Truth):")
print(df['ground_truth'].value_counts())

# 2. Confusion Matrix & Metrics
cm = confusion_matrix(df['ground_truth'], df['prediction'])
print("\nConfusion Matrix:")
print(cm)

# Metrics
acc = accuracy_score(df['ground_truth'], df['prediction'])
prec = precision_score(df['ground_truth'], df['prediction'])
rec = recall_score(df['ground_truth'], df['prediction'])
f1 = f1_score(df['ground_truth'], df['prediction'])
auc = roc_auc_score(df['ground_truth'], df['probability'])

print(f"\nPerformance Metrics:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-score: {f1:.4f}\nAUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(df['ground_truth'], df['prediction']))

# 3. Probability Distribution
plt.figure()
sns.histplot(data=df, x="probability", hue="ground_truth", kde=True, bins=25)
plt.title("Probability Distribution by Ground Truth")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. ROC Curve
fpr, tpr, _ = roc_curve(df['ground_truth'], df['probability'])
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Error Analysis
false_positives = df[(df['ground_truth'] == 0) & (df['prediction'] == 1)]
false_negatives = df[(df['ground_truth'] == 1) & (df['prediction'] == 0)]

print("\nFalse Positives:")
print(false_positives[['subject_id', 'probability']])

print("\nFalse Negatives:")
print(false_negatives[['subject_id', 'probability']])

# Optional: Save these errors
false_positives.to_csv("false_positives.csv", index=False)
false_negatives.to_csv("false_negatives.csv", index=False)
