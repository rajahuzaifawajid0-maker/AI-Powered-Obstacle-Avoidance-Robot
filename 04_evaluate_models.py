# ============================================================
# FILE 04: MODEL EVALUATION
# ============================================================
# Theory: This program evaluates all four trained models using
# Accuracy, Precision, Recall and F1 Score and creates a comparison table.

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "robot_sensor_data.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df[["left", "front", "right"]]
y = df["action"]

# Theory: Use the same split settings as training so evaluation is fair.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model_names = ["knn", "decision_tree", "svm", "random_forest"]
results = []

# Theory: Evaluate each model on the same unseen test set.
for name in model_names:
    model = joblib.load(MODEL_DIR / f"{name}.pkl")
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
    })

# Theory: Convert the results into a readable comparison table.
results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)

# Theory: Save the comparison so it can be used in the project report.
results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Theory: The first row has the highest F1 Score and is selected as the best model.
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model based on F1 Score: {best_model_name}")

# Theory: Print a confusion matrix for the best model to show which actions
# are being confused with each other.
best_model = joblib.load(MODEL_DIR / f"{best_model_name}.pkl")
best_predictions = best_model.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_predictions, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.to_csv(RESULTS_DIR / "best_model_confusion_matrix.csv")

print("\nBest model confusion matrix:")
print(cm_df)
