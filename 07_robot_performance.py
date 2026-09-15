# ============================================================
# STEP 7
# ROBOT PERFORMANCE EVALUATION
#
# This script evaluates the ML-based obstacle avoidance project.
#
# It uses:
#   - The trained KNN model
#   - The same robot sensor features
#   - The trained/tested ML performance metrics
#   - Multiple virtual robot navigation runs
#
# Output:
#   1. ML accuracy, precision, recall and F1
#   2. Robot success rate
#   3. Collision count
#   4. Completion time
#   5. Distance travelled
#   6. Average KNN inference time
#   7. CSV report saved in results/
#
# Run:
#   python 07_robot_performance.py
# ============================================================

import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "robot_sensor_data.csv"
MODEL_PATH = BASE_DIR / "models" / "knn.pkl"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

REPORT_PATH = RESULTS_DIR / "robot_performance_report.csv"


# ============================================================
# 2. CHECK FILES
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"KNN model not found:\n{MODEL_PATH}\n\n"
        "Run 03_train_models.py first."
    )


# ============================================================
# 3. LOAD DATA + MODEL
# ============================================================

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)


# ============================================================
# 4. ML MODEL EVALUATION
#
# We use the complete generated dataset here to calculate a
# simple overall model prediction summary.
#
# The official train/test comparison was already performed in
# 04_evaluate_models.py.
# ============================================================

FEATURES = ["left", "front", "right"]
TARGET = "action"

X = df[FEATURES]
y = df[TARGET]

predictions = model.predict(X)

ml_accuracy = accuracy_score(y, predictions)

ml_precision = precision_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

ml_recall = recall_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

ml_f1 = f1_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# 5. SIMULATION ENVIRONMENT
# ============================================================

ARENA_LEFT = 25
ARENA_TOP = 88
ARENA_RIGHT = 1035
ARENA_BOTTOM = 753

ROBOT_RADIUS = 27
SENSOR_RANGE = 170
SPEED = 2.9
MAX_TURN = 0.105

START = np.array([105.0, 420.0])
TARGET = np.array([955.0, 420.0])


# ============================================================
# 6. OBSTACLES
# ============================================================

OBSTACLES = [
    (250, 180, 105, 180),
    (250, 500, 105, 165),

    (440, 115, 105, 205),
    (440, 450, 105, 215),

    (630, 180, 105, 180),
    (630, 500, 105, 165),

    (820, 115, 105, 205),
    (820, 450, 105, 215),
]


# ============================================================
# 7. WAYPOINTS
# ============================================================

WAYPOINTS = [
    (185, 420),
    (185, 130),
    (400, 130),
    (400, 400),
    (400, 690),
    (590, 690),
    (590, 400),
    (590, 130),
    (780, 130),
    (780, 400),
    (780, 690),
    (980, 690),
    (980, 400),
    (955, 420),
]


# ============================================================
# 8. GEOMETRY FUNCTIONS
# ============================================================

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def point_rect_distance(x, y, rect):
    rx, ry, rw, rh = rect

    closest_x = max(rx, min(x, rx + rw))
    closest_y = max(ry, min(y, ry + rh))

    return math.hypot(
        x - closest_x,
        y - closest_y
    )


def collides(x, y):
    if x - ROBOT_RADIUS <= ARENA_LEFT:
        return True

    if x + ROBOT_RADIUS >= ARENA_RIGHT:
        return True

    if y - ROBOT_RADIUS <= ARENA_TOP:
        return True

    if y + ROBOT_RADIUS >= ARENA_BOTTOM:
        return True

    for obstacle in OBSTACLES:
        if point_rect_distance(x, y, obstacle) <= ROBOT_RADIUS:
            return True

    return False


# ============================================================
# 9. SENSOR SIMULATION
# ============================================================

def cast_sensor(x, y, angle):
    step = 3

    for distance in range(
        0,
        SENSOR_RANGE + 1,
        step
    ):
        px = x + math.cos(angle) * distance
        py = y + math.sin(angle) * distance

        if (
            px <= ARENA_LEFT
            or px >= ARENA_RIGHT
            or py <= ARENA_TOP
            or py >= ARENA_BOTTOM
        ):
            return float(distance)

        for obstacle in OBSTACLES:
            rx, ry, rw, rh = obstacle

            if (
                rx <= px <= rx + rw
                and
                ry <= py <= ry + rh
            ):
                return float(distance)

    return float(SENSOR_RANGE)


def get_sensors(x, y, angle):
    front = cast_sensor(
        x,
        y,
        angle
    )

    left = cast_sensor(
        x,
        y,
        angle - math.pi / 4
    )

    right = cast_sensor(
        x,
        y,
        angle + math.pi / 4
    )

    return left, front, right


# ============================================================
# 10. SINGLE ROBOT RUN
# ============================================================

def run_robot(max_steps=8000):
    x, y = START.copy()

    angle = 0.0

    waypoint_index = 0

    collisions = 0
    ai_decisions = 0

    distance = 0.0

    inference_times = []

    for step_number in range(max_steps):

        # ----------------------------------------------------
        # Current waypoint
        # ----------------------------------------------------
        target_x, target_y = WAYPOINTS[waypoint_index]

        waypoint_distance = math.hypot(
            target_x - x,
            target_y - y
        )

        if waypoint_distance < 18:

            if waypoint_index < len(WAYPOINTS) - 1:
                waypoint_index += 1
            else:
                return {
                    "success": True,
                    "steps": step_number,
                    "collisions": collisions,
                    "ai_decisions": ai_decisions,
                    "distance": distance,
                    "inference_ms": (
                        np.mean(inference_times)
                        if inference_times
                        else 0.0
                    ),
                }

            target_x, target_y = WAYPOINTS[waypoint_index]

        # ----------------------------------------------------
        # Sensors
        # ----------------------------------------------------
        left, front, right = get_sensors(
            x,
            y,
            angle
        )

        # ----------------------------------------------------
        # KNN prediction
        # ----------------------------------------------------
        # IMPORTANT:
        # The KNN pipeline was trained with named features.
        # Using a DataFrame with the same feature names prevents
        # the scikit-learn warning about missing feature names.
        sensor_input = pd.DataFrame(
            [[left, front, right]],
            columns=FEATURES
        )

        t0 = time.perf_counter()

        ml_action = model.predict(sensor_input)[0]

        inference_ms = (
            time.perf_counter() - t0
        ) * 1000

        inference_times.append(
            inference_ms
        )

        # ----------------------------------------------------
        # Goal direction
        # ----------------------------------------------------
        desired_angle = math.atan2(
            target_y - y,
            target_x - x
        )

        angle_error = normalize_angle(
            desired_angle - angle
        )

        # ----------------------------------------------------
        # Safety layer
        # ----------------------------------------------------
        if front < 42:

            ai_decisions += 1

            if left >= right:
                desired_angle = angle - math.pi / 2
            else:
                desired_angle = angle + math.pi / 2

            angle_error = normalize_angle(
                desired_angle - angle
            )

        elif front < 75:

            ai_decisions += 1

            if left > right + 12:
                angle_error -= 0.16

            elif right > left + 12:
                angle_error += 0.16

        # ----------------------------------------------------
        # Small ML contribution
        # ----------------------------------------------------
        if front > 75:

            if ml_action == "LEFT":
                angle_error -= 0.035

            elif ml_action == "RIGHT":
                angle_error += 0.035

        # ----------------------------------------------------
        # Smooth turn
        # ----------------------------------------------------
        turn = max(
            -MAX_TURN,
            min(MAX_TURN, angle_error)
        )

        angle = normalize_angle(
            angle + turn
        )

        # ----------------------------------------------------
        # Forward movement
        # ----------------------------------------------------
        if front < 35:
            speed = 1.0

        elif front < 60:
            speed = 2.0

        else:
            speed = SPEED

        dx = math.cos(angle) * speed
        dy = math.sin(angle) * speed

        new_x = x + dx
        new_y = y + dy

        # ----------------------------------------------------
        # Collision protection
        # ----------------------------------------------------
        if not collides(new_x, new_y):

            distance += math.hypot(
                new_x - x,
                new_y - y
            )

            x = new_x
            y = new_y

        else:

            collisions += 1
            ai_decisions += 1

            # Try turning toward the safer side.
            if left >= right:
                angle = normalize_angle(
                    angle - math.pi / 2
                )
            else:
                angle = normalize_angle(
                    angle + math.pi / 2
                )

    # --------------------------------------------------------
    # Maximum steps reached
    # --------------------------------------------------------
    return {
        "success": False,
        "steps": max_steps,
        "collisions": collisions,
        "ai_decisions": ai_decisions,
        "distance": distance,
        "inference_ms": (
            np.mean(inference_times)
            if inference_times
            else 0.0
        ),
    }



# ============================================================
# GRAPH GENERATION
# ============================================================

def create_performance_graphs(run_results, output_dir):
    """
    Create professional graphs for Step 7.

    The graphs are saved inside the results/ folder.
    They also open on screen so the results can be understood
    visually during a class presentation.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # GRAPH 1 - ML MODEL PERFORMANCE
    # --------------------------------------------------------
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    metric_values = [
        ml_accuracy * 100,
        ml_precision * 100,
        ml_recall * 100,
        ml_f1 * 100,
    ]

    plt.figure(figsize=(9, 5.5))

    bars = plt.bar(
        metric_names,
        metric_values,
    )

    plt.ylim(0, 100)
    plt.xlabel("ML Metric")
    plt.ylabel("Score (%)")
    plt.title("KNN Machine Learning Performance")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "01_ml_model_performance.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # --------------------------------------------------------
    # GRAPH 2 - ROBOT SUCCESS RATE
    # --------------------------------------------------------
    # run_results contains one dictionary per run, but the
    # dictionary itself does not store the run number.
    # Therefore, create Run 1, Run 2, ... from the list position.
    run_numbers = list(range(1, len(run_results) + 1))

    success_values = [
        100 if result["success"] else 0
        for result in run_results
    ]

    plt.figure(figsize=(9, 5.5))

    bars = plt.bar(
        run_numbers,
        success_values,
    )

    plt.ylim(0, 110)
    plt.xlabel("Simulation Run")
    plt.ylabel("Success (%)")
    plt.title("Robot Success Rate per Simulation Run")
    plt.xticks(run_numbers)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, success_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "02_robot_success_rate.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # --------------------------------------------------------
    # GRAPH 3 - COLLISIONS
    # --------------------------------------------------------
    collision_values = [
        result["collisions"]
        for result in run_results
    ]

    plt.figure(figsize=(9, 5.5))

    bars = plt.bar(
        run_numbers,
        collision_values,
    )

    plt.xlabel("Simulation Run")
    plt.ylabel("Collision Count")
    plt.title("Robot Collisions per Simulation Run")
    plt.xticks(run_numbers)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, collision_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "03_robot_collisions.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # --------------------------------------------------------
    # GRAPH 4 - KNN INFERENCE TIME
    # --------------------------------------------------------
    inference_values = [
        result["inference_ms"]
        for result in run_results
    ]

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        run_numbers,
        inference_values,
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Simulation Run")
    plt.ylabel("Inference Time (ms)")
    plt.title("KNN Inference Time per Simulation Run")
    plt.xticks(run_numbers)
    plt.grid(True, alpha=0.25)

    for x, y in zip(run_numbers, inference_values):
        plt.text(
            x,
            y + max(inference_values) * 0.03,
            f"{y:.3f} ms",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "04_knn_inference_time.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    # --------------------------------------------------------
    # GRAPH 5 - DISTANCE TRAVELLED
    # --------------------------------------------------------
    distance_values = [
        result["distance"]
        for result in run_results
    ]

    plt.figure(figsize=(9, 5.5))

    bars = plt.bar(
        run_numbers,
        distance_values,
    )

    plt.xlabel("Simulation Run")
    plt.ylabel("Distance (pixels)")
    plt.title("Robot Distance Travelled per Simulation Run")
    plt.xticks(run_numbers)
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, distance_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(distance_values) * 0.01,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    plt.savefig(
        output_dir / "05_robot_distance.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print()
    print("GRAPHS GENERATED")
    print("-" * 70)
    print("01_ml_model_performance.png")
    print("02_robot_success_rate.png")
    print("03_robot_collisions.png")
    print("04_knn_inference_time.png")
    print("05_robot_distance.png")

# ============================================================
# 11. MULTIPLE ROBOT RUNS
# ============================================================

NUMBER_OF_RUNS = 5

run_results = []

print()
print("=" * 70)
print("STEP 7 - ROBOT PERFORMANCE EVALUATION")
print("=" * 70)

print()
print("ML MODEL PERFORMANCE")
print("-" * 70)

print(
    f"Accuracy       : {ml_accuracy * 100:.2f}%"
)

print(
    f"Precision      : {ml_precision * 100:.2f}%"
)

print(
    f"Recall         : {ml_recall * 100:.2f}%"
)

print(
    f"F1 Score       : {ml_f1 * 100:.2f}%"
)

print()
print("ROBOT SIMULATION RUNS")
print("-" * 70)


for run_number in range(
    1,
    NUMBER_OF_RUNS + 1
):

    result = run_robot()

    run_results.append(result)

    print(
        f"Run {run_number}: "
        f"Success={result['success']} | "
        f"Collisions={result['collisions']} | "
        f"Distance={result['distance']:.1f}px | "
        f"Inference={result['inference_ms']:.3f}ms"
    )


# ============================================================
# 12. ROBOT METRICS
# ============================================================

success_count = sum(
    result["success"]
    for result in run_results
)

success_rate = (
    success_count / NUMBER_OF_RUNS
) * 100

average_collisions = np.mean(
    [
        result["collisions"]
        for result in run_results
    ]
)

average_distance = np.mean(
    [
        result["distance"]
        for result in run_results
    ]
)

average_ai_decisions = np.mean(
    [
        result["ai_decisions"]
        for result in run_results
    ]
)

average_inference = np.mean(
    [
        result["inference_ms"]
        for result in run_results
    ]
)


# ============================================================
# 13. CREATE REPORT
# ============================================================

report = pd.DataFrame([
    {
        "Metric": "ML Accuracy",
        "Value": round(ml_accuracy * 100, 2),
        "Unit": "%",
    },
    {
        "Metric": "ML Precision",
        "Value": round(ml_precision * 100, 2),
        "Unit": "%",
    },
    {
        "Metric": "ML Recall",
        "Value": round(ml_recall * 100, 2),
        "Unit": "%",
    },
    {
        "Metric": "ML F1 Score",
        "Value": round(ml_f1 * 100, 2),
        "Unit": "%",
    },
    {
        "Metric": "Robot Success Rate",
        "Value": round(success_rate, 2),
        "Unit": "%",
    },
    {
        "Metric": "Average Collisions",
        "Value": round(average_collisions, 2),
        "Unit": "count",
    },
    {
        "Metric": "Average Distance",
        "Value": round(average_distance, 2),
        "Unit": "pixels",
    },
    {
        "Metric": "Average AI Decisions",
        "Value": round(average_ai_decisions, 2),
        "Unit": "count",
    },
    {
        "Metric": "Average KNN Inference",
        "Value": round(average_inference, 4),
        "Unit": "ms",
    },
])


report.to_csv(
    REPORT_PATH,
    index=False
)

# Create and save professional performance graphs.
create_performance_graphs(
    run_results,
    RESULTS_DIR,
)



# ============================================================
# 14. FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL PERFORMANCE REPORT")
print("=" * 70)

print(
    f"ML Accuracy             : {ml_accuracy * 100:.2f}%"
)

print(
    f"ML Precision            : {ml_precision * 100:.2f}%"
)

print(
    f"ML Recall               : {ml_recall * 100:.2f}%"
)

print(
    f"ML F1 Score             : {ml_f1 * 100:.2f}%"
)

print(
    f"Robot Success Rate      : {success_rate:.2f}%"
)

print(
    f"Average Collisions      : {average_collisions:.2f}"
)

print(
    f"Average Distance        : {average_distance:.2f}px"
)

print(
    f"Average AI Decisions    : {average_ai_decisions:.2f}"
)

print(
    f"Average KNN Inference   : {average_inference:.4f} ms"
)

print()
print(f"Report saved to:")
print(REPORT_PATH)

print("=" * 70)
print("STEP 7 COMPLETE")
print("=" * 70)