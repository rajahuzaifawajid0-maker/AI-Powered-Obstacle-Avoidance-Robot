ML-BASED OBSTACLE AVOIDANCE ROBOT
=================================

This project uses Machine Learning to control a simulated 2D robot.
The robot receives three virtual sensor readings:
LEFT, FRONT and RIGHT obstacle distance.

The ML model predicts one of four actions:
FORWARD, LEFT, RIGHT, STOP.

FILES
-----
01_generate_dataset.py
    Generates 12,000 simulated sensor examples.

02_visualize_data.py
    Performs basic EDA and displays graphs.

03_train_models.py
    Trains KNN, Decision Tree, SVM and Random Forest.

04_evaluate_models.py
    Compares Accuracy, Precision, Recall and F1 Score.

05_test_prediction.py
    Tests a single custom sensor reading.

06_robot_simulation.py
    Opens the Pygame autonomous robot simulation.

RUN ORDER
---------
1. Install Python 3.11 or newer.
2. Open terminal inside this project folder.
3. Install libraries:
       pip install -r requirements.txt
4. Generate data:
       python 01_generate_dataset.py
5. Visualize data:
       python 02_visualize_data.py
6. Train models:
       python 03_train_models.py
7. Evaluate models:
       python 04_evaluate_models.py
8. Test a prediction:
       python 05_test_prediction.py
9. Start robot simulation:
       python 06_robot_simulation.py

IMPORTANT
---------
Run the files in the above order the first time because later files
need the dataset and trained model files created by earlier files.

The current simulation uses Random Forest as the robot brain. If you
want to use another model, change MODEL_PATH in 06_robot_simulation.py.

This is a simulation project. It does not control real hardware.
Real robot deployment would require physical sensors, a controller,
motor driver, motors and suitable safety controls.
