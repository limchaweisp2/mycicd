import pandas as pd
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def generate_data():
    """Generates synthetic classification data."""
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate():
    # 1. Generate data
    X_train, X_test, y_train, y_test = generate_data()

    # 2. Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Predict
    y_pred = clf.predict(X_test)

    # 4. Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred)
    }

    # 5. Save metrics to json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. Generate and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Training complete. Metrics saved to metrics.json and plot to confusion_matrix.png.")

if __name__ == "__main__":
    train_and_evaluate()
