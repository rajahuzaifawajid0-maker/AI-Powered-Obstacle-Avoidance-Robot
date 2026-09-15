# ============================================================
# FILE 03: TRAIN FOUR MACHINE LEARNING MODELS
# ============================================================
# Theory: This program trains KNN, Decision Tree, SVM and Random
# Forest models to predict the robot action from sensor distances.

from pathlib import Path
import joblib
import pandas as pd

# Theory: train_test_split separates training data from unseen test data.
from sklearn.model_selection import train_test_split

# Theory: StandardScaler normalizes numeric features, which is useful
# especially for distance-based KNN and SVM models.
from sklearn.preprocessing import StandardScaler

# Theory: Pipeline keeps preprocessing and model prediction together.
from sklearn.pipeline import Pipeline

# Theory: Import the four classification algorithms required by the project.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "robot_sensor_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Theory: X contains the input features given to the ML models.
X = df[["left", "front", "right"]]

# Theory: y contains the target class that the model must learn to predict.
y = df["action"]

# Theory: 80% of data is used for learning and 20% is reserved for testing.
# stratify keeps the class proportions similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Theory: KNN uses nearby training examples. Scaling makes all distance
# features comparable before KNN calculates similarity.
knn = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=7)),
])

# Theory: Decision Tree learns decision boundaries directly from sensor values.
decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)

# Theory: SVM finds boundaries between the different action classes.
# Scaling helps SVM because it is sensitive to feature magnitudes.
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", probability=True, random_state=42)),
])

# Theory: Random Forest combines many decision trees and is usually strong
# for tabular classification data.
random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)

# Theory: Store models in a dictionary so the same training code can be used
# for all four algorithms.
models = {
    "knn": knn,
    "decision_tree": decision_tree,
    "svm": svm,
    "random_forest": random_forest,
}

# Theory: Train every model on the same training data and save each trained
# model as a .pkl file for later prediction and simulation.
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    model_path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")

# Theory: Save the feature names so the simulator can provide values in the
# exact order expected by the models.
joblib.dump(["left", "front", "right"], MODEL_DIR / "feature_names.pkl")

print("\nAll four models trained and saved successfully.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
