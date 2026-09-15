# ============================================================
# ML-BASED OBSTACLE AVOIDANCE ROBOT
# FILE: 01_generate_dataset.py
# ============================================================
#
# THEORY:
# This file creates a simulated dataset for our robot.
#
# We do not have real physical sensors at this stage.
# Therefore, we simulate three distance sensors:
#
# 1. LEFT SENSOR  -> distance of obstacle on the left
# 2. FRONT SENSOR -> distance of obstacle in front
# 3. RIGHT SENSOR -> distance of obstacle on the right
#
# These three sensor values will become the INPUT (X)
# of our Machine Learning models.
#
# The robot action will become the TARGET (y).
#
# Possible actions:
#     FORWARD
#     LEFT
#     RIGHT
#     STOP
#
# Complete ML pipeline:
#
# Sensor Data
#      ↓
# Dataset Generation
#      ↓
# Data Preprocessing
#      ↓
# Machine Learning
#      ↓
# Robot Action
#
# IMPORTANT:
# We are generating many different sensor situations
# instead of using only a few fixed values.
# This helps the ML model learn general patterns.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# THEORY:
# NumPy is used to generate random numerical sensor values.
# We use it because simulated sensors should produce
# different readings instead of always returning the same value.
import numpy as np


# THEORY:
# Pandas is used to create and save the dataset.
# It stores our data in a table-like DataFrame.
import pandas as pd


# THEORY:
# Path is used to create file/folder paths safely.
# It works properly on Windows, Linux and other systems.
from pathlib import Path


# ============================================================
# 2. SET RANDOM SEED
# ============================================================

# THEORY:
# Random numbers are used in this project to simulate
# different sensor readings.
#
# random seed makes the random generation reproducible.
# This means that if we run the program again,
# we can generate the same dataset pattern.
np.random.seed(42)


# ============================================================
# 3. PROJECT PATH
# ============================================================

# THEORY:
# BASE_DIR represents the main project folder.
#
# __file__ gives the location of this Python file.
# .resolve() converts it into the complete absolute path.
BASE_DIR = Path(__file__).resolve().parent


# THEORY:
# DATA_DIR represents the folder where our dataset will be saved.
DATA_DIR = BASE_DIR / "data"


# THEORY:
# Create the data folder if it does not already exist.
#
# exist_ok=True means:
# If the folder already exists, Python will not produce an error.
DATA_DIR.mkdir(parents=True, exist_ok=True)


# THEORY:
# This is the final CSV file where our generated
# robot sensor dataset will be stored.
OUTPUT_FILE = DATA_DIR / "robot_sensor_data.csv"


# ============================================================
# 4. DATASET SIZE
# ============================================================

# THEORY:
# We will generate 12,000 sensor examples.
#
# A large number of examples gives the ML algorithms
# more situations from which to learn.
TOTAL_ROWS = 12000


# ============================================================
# 5. SENSOR RANGE
# ============================================================

# THEORY:
# Our virtual sensors can detect obstacles between
# 5 cm and 120 cm.
#
# 5 cm  -> obstacle is very close
# 120 cm -> obstacle is relatively far away
MIN_DISTANCE = 5.0
MAX_DISTANCE = 120.0


# ============================================================
# 6. NUMBER OF EXAMPLES FOR EACH ACTION
# ============================================================

# THEORY:
# Machine Learning works better when the target classes
# are reasonably balanced.
#
# In our previous dataset:
#
# FORWARD = 8292
# LEFT    = 1853
# RIGHT   = 1811
# STOP    = 44
#
# STOP was extremely small.
#
# Therefore, this final version intentionally creates
# approximately equal numbers of examples for each action.
#
# 12000 total rows / 4 classes = 3000 examples per class.

ROWS_PER_CLASS = TOTAL_ROWS // 4


# ============================================================
# 7. FUNCTION FOR SENSOR VALUES
# ============================================================

def random_distance(size):
    """
    THEORY:
    This function generates random virtual sensor distances.

    Each value will be between 5 cm and 120 cm.

    INPUT:
        size -> number of sensor readings required

    OUTPUT:
        NumPy array containing random distances
    """

    # THEORY:
    # np.random.uniform generates decimal random values
    # between MIN_DISTANCE and MAX_DISTANCE.
    return np.random.uniform(
        MIN_DISTANCE,
        MAX_DISTANCE,
        size
    )


# ============================================================
# 8. GENERATE FORWARD SCENARIOS
# ============================================================

def generate_forward_data(n):
    """
    THEORY:
    This function creates situations where the robot
    should move FORWARD.

    For FORWARD:
    The front distance should generally be safe/far.
    """

    # THEORY:
    # Generate random left sensor readings.
    left = random_distance(n)

    # THEORY:
    # Generate random right sensor readings.
    right = random_distance(n)

    # THEORY:
    # Front distance should normally be relatively large
    # because the robot has enough space to move forward.
    front = np.random.uniform(55, MAX_DISTANCE, n)

    # THEORY:
    # Create the target label.
    # Every generated example in this function represents
    # a FORWARD decision.
    action = np.full(n, "FORWARD")

    # THEORY:
    # Return all four columns.
    return left, front, right, action


# ============================================================
# 9. GENERATE LEFT-TURN SCENARIOS
# ============================================================

def generate_left_data(n):
    """
    THEORY:
    This function creates situations where the robot
    should turn LEFT.

    A left turn is useful when:
    - The front is blocked
    - The right side is more open
    - The left side is reasonably available
    """

    # THEORY:
    # Left side has enough space for the robot.
    left = np.random.uniform(40, MAX_DISTANCE, n)

    # THEORY:
    # Front has an obstacle at a closer distance.
    front = np.random.uniform(5, 45, n)

    # THEORY:
    # Right side can also have different distances,
    # but it is generally less preferred than the left.
    right = np.random.uniform(20, 80, n)

    # THEORY:
    # Assign LEFT as the target action.
    action = np.full(n, "LEFT")

    # THEORY:
    # Return sensor data and target label.
    return left, front, right, action


# ============================================================
# 10. GENERATE RIGHT-TURN SCENARIOS
# ============================================================

def generate_right_data(n):
    """
    THEORY:
    This function creates situations where the robot
    should turn RIGHT.

    A right turn is useful when:
    - The front is blocked
    - The right side has enough free space
    """

    # THEORY:
    # Left side is relatively less preferred.
    left = np.random.uniform(20, 80, n)

    # THEORY:
    # Front obstacle is close.
    front = np.random.uniform(5, 45, n)

    # THEORY:
    # Right side has enough space for turning.
    right = np.random.uniform(40, MAX_DISTANCE, n)

    # THEORY:
    # Assign RIGHT as target action.
    action = np.full(n, "RIGHT")

    # THEORY:
    # Return generated values.
    return left, front, right, action


# ============================================================
# 11. GENERATE STOP SCENARIOS
# ============================================================

def generate_stop_data(n):
    """
    THEORY:
    This function creates dangerous situations
    where the robot should STOP.

    Example:
        Left  = 10 cm
        Front = 8 cm
        Right = 12 cm

    All directions are blocked.
    Therefore, STOP is the safest action.
    """

    # THEORY:
    # Left obstacle is very close.
    left = np.random.uniform(5, 25, n)

    # THEORY:
    # Front obstacle is very close.
    front = np.random.uniform(5, 20, n)

    # THEORY:
    # Right obstacle is also very close.
    right = np.random.uniform(5, 25, n)

    # THEORY:
    # Assign STOP as the target action.
    action = np.full(n, "STOP")

    # THEORY:
    # Return generated values.
    return left, front, right, action


# ============================================================
# 12. GENERATE ALL FOUR CLASSES
# ============================================================

# THEORY:
# Generate FORWARD examples.
forward_data = generate_forward_data(ROWS_PER_CLASS)


# THEORY:
# Generate LEFT examples.
left_data = generate_left_data(ROWS_PER_CLASS)


# THEORY:
# Generate RIGHT examples.
right_data = generate_right_data(ROWS_PER_CLASS)


# THEORY:
# Generate STOP examples.
stop_data = generate_stop_data(ROWS_PER_CLASS)


# ============================================================
# 13. COMBINE ALL SENSOR DATA
# ============================================================

# THEORY:
# Concatenate means joining the data from all four
# action classes into one large array.
#
# Result:
# 3000 FORWARD
# 3000 LEFT
# 3000 RIGHT
# 3000 STOP
#
# Total = 12000 rows.

left_values = np.concatenate([
    forward_data[0],
    left_data[0],
    right_data[0],
    stop_data[0]
])


# THEORY:
# Combine all FRONT sensor values.
front_values = np.concatenate([
    forward_data[1],
    left_data[1],
    right_data[1],
    stop_data[1]
])


# THEORY:
# Combine all RIGHT sensor values.
right_values = np.concatenate([
    forward_data[2],
    left_data[2],
    right_data[2],
    stop_data[2]
])


# THEORY:
# Combine all target action labels.
actions = np.concatenate([
    forward_data[3],
    left_data[3],
    right_data[3],
    stop_data[3]
])


# ============================================================
# 14. CREATE DATAFRAME
# ============================================================

# THEORY:
# A DataFrame is a table structure provided by Pandas.
#
# Our table has:
#     left
#     front
#     right
#     action
#
# The first three are INPUT FEATURES.
# The last one is the TARGET.

df = pd.DataFrame({
    "left": left_values,
    "front": front_values,
    "right": right_values,
    "action": actions
})


# ============================================================
# 15. SHUFFLE THE DATASET
# ============================================================

# THEORY:
# Currently, our data is arranged like this:
#
# FORWARD
# FORWARD
# FORWARD
# ...
# LEFT
# LEFT
# LEFT
# ...
# RIGHT
# RIGHT
# RIGHT
# ...
# STOP
# STOP
# STOP
#
# This order is not ideal for ML training.
#
# Therefore, we randomly shuffle the rows.
#
# random_state=42 makes the shuffle reproducible.

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 16. ROUND SENSOR VALUES
# ============================================================

# THEORY:
# Sensor values do not need extremely long decimal values.
#
# Example:
# 75.34892764
#
# becomes:
# 75.35
#
# This makes the dataset easier to read.

df["left"] = df["left"].round(2)
df["front"] = df["front"].round(2)
df["right"] = df["right"].round(2)


# ============================================================
# 17. SAVE DATASET
# ============================================================

# THEORY:
# to_csv() saves the DataFrame as a CSV file.
#
# index=False prevents Pandas from creating an
# unnecessary extra index column.

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

# THEORY:
# Print a success message so the user knows
# that dataset generation completed successfully.

print("=" * 60)
print("ML-BASED OBSTACLE AVOIDANCE ROBOT")
print("DATASET GENERATION COMPLETE")
print("=" * 60)


# THEORY:
# Display the exact location of the generated CSV file.
print(f"File: {OUTPUT_FILE}")


# THEORY:
# Display number of rows and columns.
print(f"Dataset shape: {df.shape}")


# THEORY:
# Display the distribution of robot actions.
#
# This is especially important because we want
# the four classes to be reasonably balanced.

print("\nClass distribution:")
print(df["action"].value_counts())


# THEORY:
# Display the first 10 rows so we can verify
# that the dataset looks correct.

print("\nFirst 10 rows:")
print(df.head(10))


# ============================================================
# 19. FINAL DATA VALIDATION
# ============================================================

# THEORY:
# These checks confirm that no sensor value is outside
# our allowed range.
#
# If everything is correct, all results should be True.

print("\nSensor range validation:")

print(
    "Left sensor valid:",
    df["left"].between(
        MIN_DISTANCE,
        MAX_DISTANCE
    ).all()
)

print(
    "Front sensor valid:",
    df["front"].between(
        MIN_DISTANCE,
        MAX_DISTANCE
    ).all()
)

print(
    "Right sensor valid:",
    df["right"].between(
        MIN_DISTANCE,
        MAX_DISTANCE
    ).all()
)


# THEORY:
# Check for missing values.
#
# The result should be 0 for every column.

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# END OF FILE
# ============================================================
#
# FINAL OUTPUT:
#
# data/
#     robot_sensor_data.csv
#
# This CSV will later be used by:
#
# 02_visualize_data.py
#          ↓
# 03_train_models.py
#          ↓
# 04_evaluate_models.py
#          ↓
# 05_test_prediction.py
#          ↓
# 06_robot_simulation.py
#
# ============================================================

