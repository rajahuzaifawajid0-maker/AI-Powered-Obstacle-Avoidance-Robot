# ============================================================
# FILE 02: DATA VISUALIZATION / EDA
# ============================================================
# Theory: This program explores the generated sensor dataset
# before machine learning training starts.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Theory: Find the project folder automatically.
BASE_DIR = Path(__file__).resolve().parent

# Theory: Read the CSV created by file 01.
DATA_PATH = BASE_DIR / "data" / "robot_sensor_data.csv"

df = pd.read_csv(DATA_PATH)

# Theory: Print dataset shape so we know rows and columns.
print("Dataset shape:", df.shape)

# Theory: Print basic statistics for the three sensor features.
print("\nSensor statistics:")
print(df[["left", "front", "right"]].describe())

# Theory: Check whether any missing values exist.
print("\nMissing values:")
print(df.isnull().sum())

# Theory: Show how many examples belong to each action class.
print("\nAction distribution:")
print(df["action"].value_counts())

# Theory: Create a bar chart showing class distribution.
df["action"].value_counts().plot(kind="bar")
plt.title("Robot Action Distribution")
plt.xlabel("Action")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()

# Theory: Create a scatter plot to see how front distance and left distance
# relate to robot actions.
for action in df["action"].unique():
    subset = df[df["action"] == action]
    plt.scatter(subset["front"], subset["left"], label=action, alpha=0.35)

plt.title("Front Distance vs Left Distance")
plt.xlabel("Front Distance (cm)")
plt.ylabel("Left Distance (cm)")
plt.legend()
plt.tight_layout()
plt.show()
