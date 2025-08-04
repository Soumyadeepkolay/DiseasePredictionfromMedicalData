# Code For Disease Prediction------>
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.datasets import load_breast_cancer, load_diabetes

# Function to load datasets based on selection
def load_dataset(name):
    # if name == "Heart Disease":
    #     data = pd.read_csv('heart.csv')  # Uncomment if Heart dataset is available
    #     X = data.drop('target', axis=1)
    #     y = data['target']

    if name == "Breast Cancer":
        dataset = load_breast_cancer()
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.Series(dataset.target)

    elif name == "Diabetes":
        dataset = load_diabetes()
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        # Convert regression target to binary classification
        y = pd.Series((dataset.target > dataset.target.mean()).astype(int))

    else:
        raise ValueError("Invalid dataset name.")
    
    return X, y

# List of datasets to run
datasets = ["Heart Disease", "Breast Cancer", "Diabetes"]

# Define models
models = {
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Loop through each dataset
for dataset_name in datasets:
    print(f"\n{'='*50}\nDataset: {dataset_name}\n{'='*50}")

    try:
        # Load data
        X, y = load_dataset(dataset_name)
    except ValueError as e:
        print(f"Skipping dataset: {e}")
        continue

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Plot ROC Curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # Finalize ROC Plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {dataset_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Optional: Feature Importance for Random Forest and Gradient Boosting
    for model_name in ['Random Forest', 'Gradient Boosting']:
        model = models[model_name]
        importances = model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances[indices], y=features[indices])
        plt.title(f"Feature Importance - {model_name} ({dataset_name})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    # Cross-validation on Logistic Regression
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
    print("\nLogistic Regression 5-Fold CV Accuracy: %.4f ± %.4f" % (cv_scores.mean(), cv_scores.std()))

print("\n All models evaluated on all datasets successfully.")