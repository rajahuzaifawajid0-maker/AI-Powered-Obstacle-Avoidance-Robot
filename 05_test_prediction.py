# ============================================================
# FILE 05: TEST A SINGLE ROBOT SENSOR PREDICTION
# ============================================================
# Theory: This program loads the trained models and predicts an action
# for a new sensor reading entered by the user.

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Theory: Use the Random Forest model as the default prediction model.
MODEL_PATH = MODEL_DIR / "random_forest.pkl"
model = joblib.load(MODEL_PATH)

# Theory: Ask the user for three virtual sensor distances.
left = float(input("Enter LEFT distance (cm): "))
front = float(input("Enter FRONT distance (cm): "))
right = float(input("Enter RIGHT distance (cm): "))

# Theory: Put the sensor values into a DataFrame with the exact feature names
# used during training.
sensor_data = pd.DataFrame([[left, front, right]], columns=["left", "front", "right"])

# Theory: The trained model predicts the robot's next action.
prediction = model.predict(sensor_data)[0]

print(f"\nSensor values: Left={left}, Front={front}, Right={right}")
print(f"Predicted action: {prediction}")

# Theory: If the model supports probabilities, show confidence-like class scores.
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sensor_data)[0]
    classes = model.classes_
    probability_table = sorted(
        zip(classes, probabilities), key=lambda item: item[1], reverse=True
    )
    print("\nClass probabilities:")
    for action, probability in probability_table:
        print(f"{action:10s}: {probability * 100:.2f}%")
