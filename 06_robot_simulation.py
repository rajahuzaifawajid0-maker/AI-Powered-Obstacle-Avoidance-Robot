# ============================================================
# ML-BASED OBSTACLE AVOIDANCE ROBOT
# PROFESSIONAL CLASSROOM DEMO
#
# ML MODEL:
#   KNN
#
# NAVIGATION:
#   Reactive "widest open space" scanning. Every frame the robot
#   fans out a wide arc of sensor rays around its heading, scores
#   each direction by how much open space it has (radius-aware,
#   so narrow gaps it cannot actually fit through are correctly
#   treated as blocked) and by how well it points toward the
#   goal, and steers toward the best one. There is no scripted
#   route -- the exact path emerges live. A stuck/oscillation
#   watchdog forces a decisive escape turn if the robot ever
#   stalls against geometry, so it can never get trapped spinning
#   in one spot.
#
# FEATURES:
#   1. Professional robot body + two wheels
#   2. 3 ultrasonic-style sensor beams (fed to the KNN model)
#   3. Wide radius-aware scanning fan for real navigation
#   4. Smooth autonomous movement, no scripted route
#   5. KNN prediction shown live and blended into steering
#   6. Stuck / local-minima watchdog with forced escape
#   7. Target / destination
#   8. Robot trail
#   9. Dashboard-style HUD, 1920x1080
#  10. Speed, distance, collision and inference metrics
#  11. Start / pause / reset controls
#
# RUN:
#   python 06_robot_simulation.py
# ============================================================

import math
import random
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pygame


# ============================================================
# 1. PROJECT PATH + MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "knn.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"KNN model not found:\n{MODEL_PATH}\n\n"
        "Run 03_train_models.py first."
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# 2. PYGAME
# ============================================================

pygame.init()

# --------------------------------------------------------------
# Fit the window to the actual screen instead of hard-coding
# 1920x1080. On many laptops, Windows display scaling and the
# taskbar mean a literal 1920x1080 window renders larger than
# the visible area and gets clipped (the sidebar disappearing
# off the edge). We ask the OS for the real usable screen size
# and cap the window to fit inside it with a safety margin.
# --------------------------------------------------------------
_display_info = pygame.display.Info()
_screen_w = _display_info.current_w
_screen_h = _display_info.current_h

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

WIDTH = min(TARGET_WIDTH, _screen_w - 16)
HEIGHT = min(TARGET_HEIGHT, _screen_h - 80)  # leave room for taskbar/title bar

# UI_SCALE compresses fonts and spacing on any screen shorter than a
# full 1080px so the sidebar's content always fits without overlapping
# (Windows display scaling can report a smaller usable height than the
# monitor's physical resolution).
UI_SCALE = max(0.70, min(1.0, HEIGHT / 1080))


def sc(value):
    """Scale a spacing/size value by UI_SCALE."""
    return int(round(value * UI_SCALE))


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Robot Lab | ML-Based Obstacle Avoidance")

clock = pygame.time.Clock()


# ============================================================
# 3. FONTS
# ============================================================

FONT = pygame.font.SysFont("segoeui", sc(22))
SMALL = pygame.font.SysFont("segoeui", sc(18))
TINY = pygame.font.SysFont("segoeui", sc(15))
TITLE = pygame.font.SysFont("segoeui", sc(34), bold=True)
BIG = pygame.font.SysFont("segoeui", sc(42), bold=True)
MONO = pygame.font.SysFont("consolas", sc(18))
MONO_SMALL = pygame.font.SysFont("consolas", sc(15))


# ============================================================
# 4. PROFESSIONAL COLOR PALETTE
# ============================================================

BACKGROUND = (9, 12, 18)
PANEL = (16, 22, 32)
PANEL_2 = (21, 29, 41)
GRID = (28, 38, 52)

WHITE = (233, 240, 248)
MUTED = (128, 142, 162)
FAINT = (72, 84, 102)

CYAN = (34, 211, 238)
BLUE = (59, 130, 246)
GREEN = (52, 211, 153)
YELLOW = (250, 204, 21)
RED = (248, 81, 73)
ORANGE = (251, 146, 60)
PURPLE = (167, 139, 250)
GOLD = (223, 184, 92)

OBSTACLE = (52, 59, 72)
OBSTACLE_EDGE = (94, 105, 122)

ROAD = (13, 17, 25)
TRAIL = (56, 189, 248)

TARGET_COLOR = (52, 211, 153)
TARGET_RING = (110, 245, 190)


# ============================================================
# 5. LAYOUT  (1920 x 1080)
# ============================================================

TOP_BAR = pygame.Rect(0, 0, WIDTH, 92)

SIDEBAR_WIDTH = 380

ARENA = pygame.Rect(
    30,
    118,
    WIDTH - SIDEBAR_WIDTH - 30 - 20 - 30,
    HEIGHT - 118 - 30
)

SIDEBAR = pygame.Rect(
    ARENA.right + 20,
    118,
    SIDEBAR_WIDTH,
    HEIGHT - 118 - 30
)


# ============================================================
# 6. ROBOT SETTINGS
# ============================================================

ROBOT_LENGTH = 58
ROBOT_WIDTH = 42
ROBOT_RADIUS = 30

SENSOR_RANGE = 210

BASE_SPEED = 3.3
MAX_TURN = 0.115


# ============================================================
# 7. OBSTACLES
#
# Four column-pairs spanning the arena. Each column's gap is
# offset alternately high / low (a zigzag corridor), so there is
# no straight-line shot from start to target -- the robot has to
# actually weave up and around each obstacle pair to get through,
# which is what makes its live scanning/avoidance visible. Every
# gap is still well over twice the robot's radius, so a route
# always exists.
#
# The gap centers are computed once here so both the obstacles
# AND the start/target positions can align with them -- placing
# the target in the open lane instead of near an obstacle corner
# is what keeps the final approach clean instead of forcing the
# robot to cut a tight corner right next to the goal.
# ============================================================

ZIGZAG_COL_FRACTIONS = (0.17, 0.39, 0.61, 0.83)
ZIGZAG_HIGH_FRAC = 0.30
ZIGZAG_LOW_FRAC = 0.70


def zigzag_gap_center(index):
    frac = ZIGZAG_HIGH_FRAC if index % 2 == 0 else ZIGZAG_LOW_FRAC
    return ARENA.top + ARENA.height * frac


def make_obstacles():
    obstacles = []

    col_width = max(70, min(150, int(ARENA.width * 0.075)))
    gap_half = max(80, int(ARENA.height * 0.10))

    margin_top = ARENA.top + 60
    margin_bottom = ARENA.bottom - 60

    xs = [ARENA.left + ARENA.width * frac for frac in ZIGZAG_COL_FRACTIONS]

    for i, x in enumerate(xs):
        gap_center = zigzag_gap_center(i)

        top_h = (gap_center - gap_half) - margin_top
        bottom_y = gap_center + gap_half
        bottom_h = margin_bottom - bottom_y

        if top_h > 40:
            obstacles.append(pygame.Rect(x, margin_top, col_width, top_h))

        if bottom_h > 40:
            obstacles.append(pygame.Rect(x, bottom_y, col_width, bottom_h))

    return obstacles


OBSTACLES = make_obstacles()

# Start sits in the open lane before the first obstacle; target
# sits in the open lane after the last obstacle -- both aligned
# with that column's actual gap band, so neither end forces the
# robot to cut a corner right at the start or the finish.
START = pygame.Vector2(ARENA.left + 100, zigzag_gap_center(0))
TARGET_POS = pygame.Vector2(
    ARENA.right - 100, zigzag_gap_center(len(ZIGZAG_COL_FRACTIONS) - 1)
)


# ============================================================
# 8. STATE
#
# No fixed waypoints. The robot only knows the final TARGET_POS
# and finds its own way there using its sensors.
# ============================================================

robot_pos = START.copy()
robot_angle = 0.0

running = True
paused = False
finished = False

collision_count = 0
ai_decisions = 0
prediction_count = 0

total_distance = 0.0
last_action = "FORWARD"

path_points = []

simulation_start = time.perf_counter()
completion_time = None

last_sensors = (SENSOR_RANGE, SENSOR_RANGE, SENSOR_RANGE)

# --- open-space scanning state ---
NUM_SCAN_RAYS = 15
SCAN_SPREAD = math.radians(180)
SCAN_OFFSETS = [
    -SCAN_SPREAD / 2 + i * (SCAN_SPREAD / (NUM_SCAN_RAYS - 1))
    for i in range(NUM_SCAN_RAYS)
]

smoothed_offset = 0.0
last_scan = []
chosen_offset = 0.0
open_space_reading = SENSOR_RANGE

# --- stuck / oscillation watchdog ---
POSITION_HISTORY_LEN = 75
position_history = deque(maxlen=POSITION_HISTORY_LEN)
escape_mode = False
escape_timer = 0
escape_turn_dir = 1.0


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def point_rect_distance(point, rect):
    x = clamp(point.x, rect.left, rect.right)
    y = clamp(point.y, rect.top, rect.bottom)

    return math.hypot(point.x - x, point.y - y)


def inside_arena(pos):
    return (
        ARENA.left + ROBOT_RADIUS < pos.x < ARENA.right - ROBOT_RADIUS
        and
        ARENA.top + ROBOT_RADIUS < pos.y < ARENA.bottom - ROBOT_RADIUS
    )


def collides(pos):
    if not inside_arena(pos):
        return True

    for obstacle in OBSTACLES:
        if point_rect_distance(pos, obstacle) <= ROBOT_RADIUS:
            return True

    return False


# ============================================================
# 10. SENSOR SYSTEM
# ============================================================

def cast_sensor(origin, angle):
    """Literal ray distance -- what the KNN model was trained on."""
    step = 3

    for distance in range(0, SENSOR_RANGE + 1, step):
        x = origin.x + math.cos(angle) * distance
        y = origin.y + math.sin(angle) * distance

        if (
            x <= ARENA.left
            or x >= ARENA.right
            or y <= ARENA.top
            or y >= ARENA.bottom
        ):
            return float(distance)

        for obstacle in OBSTACLES:
            if obstacle.collidepoint(x, y):
                return float(distance)

    return float(SENSOR_RANGE)


def cast_sensor_safe(origin, angle):
    """Radius-aware ray distance -- takes the robot's own body
    size into account, so a gap the robot physically cannot fit
    through reads as blocked instead of looking falsely open.
    Used only for navigation decisions, never for the ML model."""
    step = 4

    for distance in range(0, SENSOR_RANGE + 1, step):
        x = origin.x + math.cos(angle) * distance
        y = origin.y + math.sin(angle) * distance
        point = pygame.Vector2(x, y)

        if (
            x <= ARENA.left + ROBOT_RADIUS
            or x >= ARENA.right - ROBOT_RADIUS
            or y <= ARENA.top + ROBOT_RADIUS
            or y >= ARENA.bottom - ROBOT_RADIUS
        ):
            return float(distance)

        for obstacle in OBSTACLES:
            if point_rect_distance(point, obstacle) <= ROBOT_RADIUS:
                return float(distance)

    return float(SENSOR_RANGE)


def get_sensors():
    """The 3 sensors the KNN model was trained on."""
    front = cast_sensor(robot_pos, robot_angle)
    left = cast_sensor(robot_pos, robot_angle - math.pi / 4)
    right = cast_sensor(robot_pos, robot_angle + math.pi / 4)

    return left, front, right


# ============================================================
# 11. KNN AI
# ============================================================

def predict_action(left, front, right):
    global prediction_count

    start = time.perf_counter()

    values = np.array([[left, front, right]], dtype=float)
    prediction = model.predict(values)[0]

    prediction_count += 1

    inference_ms = (time.perf_counter() - start) * 1000

    return str(prediction), inference_ms


# ============================================================
# 12. OPEN-SPACE SCANNING
# ============================================================

def scan_open_space():
    """Fan out rays around the heading (radius-aware) and return
    every (offset, distance) reading, plus the best direction:
    the one with the most real, fittable open space, weighted
    gently toward the target and toward the current heading."""
    to_target = TARGET_POS - robot_pos
    target_angle = math.atan2(to_target.y, to_target.x)
    target_offset = normalize_angle(target_angle - robot_angle)

    readings = []
    best_offset = 0.0
    best_score = -1e9

    for offset in SCAN_OFFSETS:
        angle = robot_angle + offset
        dist = cast_sensor_safe(robot_pos, angle)
        readings.append((offset, dist))

        angle_diff_to_target = abs(normalize_angle(offset - target_offset))

        score = (
            dist
            - angle_diff_to_target * 60.0
            - abs(offset) * 5.0
        )

        if score > best_score:
            best_score = score
            best_offset = offset

    chosen_dist = min(
        (d for o, d in readings if abs(o - best_offset) < 1e-6),
        default=SENSOR_RANGE
    )

    return best_offset, chosen_dist, readings


# ============================================================
# 13. NAVIGATION / MOVEMENT
# ============================================================

def update_robot():
    global robot_pos
    global robot_angle
    global collision_count
    global ai_decisions
    global total_distance
    global last_action
    global last_sensors
    global finished
    global completion_time
    global smoothed_offset
    global last_scan
    global chosen_offset
    global open_space_reading
    global escape_mode
    global escape_timer
    global escape_turn_dir

    left, front, right = get_sensors()
    last_sensors = (left, front, right)

    ml_action, inference_ms = predict_action(left, front, right)

    if finished:
        return ml_action, inference_ms

    distance_to_target = robot_pos.distance_to(TARGET_POS)

    if distance_to_target < 34:
        finished = True
        completion_time = time.perf_counter() - simulation_start
        return "FINISHED", inference_ms

    # --------------------------------------------------------
    # Stuck / oscillation watchdog: if the robot hasn't gone
    # anywhere meaningful over the last couple of seconds, it
    # is wedged or looping. Force a decisive turn+retreat until
    # it has clearly broken free, then resume normal scanning.
    # Skipped when already very close to the goal -- final
    # approach naturally involves small slow movements that
    # should not be mistaken for being stuck.
    # --------------------------------------------------------
    position_history.append(robot_pos.copy())

    if (
        not escape_mode
        and distance_to_target > 90
        and len(position_history) == POSITION_HISTORY_LEN
    ):
        spread = max(
            p.distance_to(position_history[0]) for p in position_history
        )
        if spread < 40:
            escape_mode = True
            escape_timer = 0
            escape_turn_dir = -1.0 if left > right else 1.0
            ai_decisions += 1

    if escape_mode:
        escape_timer += 1

        turn = escape_turn_dir * MAX_TURN
        robot_angle = normalize_angle(robot_angle + turn)

        speed = 1.6
        movement = pygame.Vector2(
            math.cos(robot_angle), math.sin(robot_angle)
        ) * speed

        proposed = robot_pos + movement
        if not collides(proposed):
            total_distance += robot_pos.distance_to(proposed)
            robot_pos = proposed

        if escape_timer > 55 and robot_pos.distance_to(position_history[0]) > 70:
            escape_mode = False
            position_history.clear()
        elif escape_timer > 130:
            escape_mode = False
            position_history.clear()

        last_action = ml_action
        return ml_action, inference_ms

    # --------------------------------------------------------
    # Normal mode: scan for the most open, goal-aligned direction.
    # --------------------------------------------------------
    best_offset, chosen_dist, readings = scan_open_space()

    last_scan = readings
    chosen_offset = best_offset
    open_space_reading = chosen_dist

    if chosen_dist < 85:
        ai_decisions += 1

    smoothed_offset = smoothed_offset * 0.82 + best_offset * 0.18

    turn_target = smoothed_offset

    if front > 90:
        if ml_action == "LEFT":
            turn_target -= 0.035
        elif ml_action == "RIGHT":
            turn_target += 0.035

    turn = clamp(turn_target, -MAX_TURN, MAX_TURN)
    robot_angle = normalize_angle(robot_angle + turn)

    speed = BASE_SPEED
    if front < 40:
        speed = 1.1
    elif front < 75 or chosen_dist < 75:
        speed = 2.1

    movement = pygame.Vector2(
        math.cos(robot_angle), math.sin(robot_angle)
    ) * speed

    proposed = robot_pos + movement

    if not collides(proposed):
        total_distance += robot_pos.distance_to(proposed)
        robot_pos = proposed
    else:
        collision_count += 1
        ai_decisions += 1

        side_angles = [
            robot_angle - math.pi / 2,
            robot_angle + math.pi / 2,
        ]
        if right > left:
            side_angles.reverse()

        for angle in side_angles:
            escape = pygame.Vector2(
                math.cos(angle), math.sin(angle)
            ) * 2.2

            candidate = robot_pos + escape
            if not collides(candidate):
                robot_pos = candidate
                robot_angle = angle
                break

    last_action = ml_action

    return ml_action, inference_ms


# ============================================================
# 14. DRAW GRID
# ============================================================

def draw_grid():
    pygame.draw.rect(screen, ROAD, ARENA, border_radius=16)

    for x in range(ARENA.left + 30, ARENA.right, 46):
        pygame.draw.line(screen, GRID, (x, ARENA.top), (x, ARENA.bottom), 1)

    for y in range(ARENA.top + 30, ARENA.bottom, 46):
        pygame.draw.line(screen, GRID, (ARENA.left, y), (ARENA.right, y), 1)

    pygame.draw.rect(screen, (58, 72, 94), ARENA, 2, border_radius=16)


# ============================================================
# 15. DRAW OBSTACLES
# ============================================================

def draw_obstacles():
    for obstacle in OBSTACLES:
        shadow = obstacle.move(6, 8)
        pygame.draw.rect(screen, (7, 10, 15), shadow, border_radius=11)
        pygame.draw.rect(screen, OBSTACLE, obstacle, border_radius=11)
        pygame.draw.rect(screen, OBSTACLE_EDGE, obstacle, 2, border_radius=11)

        for y in range(obstacle.top + 16, obstacle.bottom - 6, 30):
            pygame.draw.line(
                screen,
                (98, 108, 126),
                (obstacle.left + 14, y),
                (obstacle.right - 14, y),
                2
            )


# ============================================================
# 16. TARGET
# ============================================================

def draw_target():
    center = (int(TARGET_POS.x), int(TARGET_POS.y))

    pulse = 4 * math.sin(time.perf_counter() * 2.4)

    pygame.draw.circle(screen, (30, 78, 58), center, int(42 + pulse), 2)
    pygame.draw.circle(screen, TARGET_RING, center, 33, 2)
    pygame.draw.circle(screen, TARGET_COLOR, center, 22)
    pygame.draw.circle(screen, BACKGROUND, center, 9)

    pygame.draw.line(
        screen, TARGET_RING,
        (center[0] - 11, center[1]), (center[0] + 11, center[1]), 2
    )
    pygame.draw.line(
        screen, TARGET_RING,
        (center[0], center[1] - 11), (center[0], center[1] + 11), 2
    )

    text("TARGET", TARGET_POS.x - 34, TARGET_POS.y + 42, SMALL, TARGET_COLOR)


# ============================================================
# 17. PATH / TRAIL
#
# Shows the actual route the robot discovered on its own this
# run -- it will differ run to run since nothing is scripted.
# ============================================================

def draw_path():
    if len(path_points) < 2:
        return

    points = [(int(p.x), int(p.y)) for p in path_points]
    pygame.draw.lines(screen, TRAIL, False, points, 3)


# ============================================================
# 18. OPEN-SPACE SCAN FAN
# ============================================================

def draw_scan_fan(readings, best_offset):
    for offset, dist in readings:
        angle = robot_angle + offset
        end = robot_pos + pygame.Vector2(
            math.cos(angle), math.sin(angle)
        ) * dist

        is_chosen = abs(offset - best_offset) < 1e-6

        if is_chosen:
            color = YELLOW
            width = 3
        else:
            fade = clamp(dist / SENSOR_RANGE, 0.15, 0.55)
            color = (
                int(70 * fade + 18),
                int(92 * fade + 24),
                int(114 * fade + 34),
            )
            width = 1

        pygame.draw.line(
            screen, color,
            (int(robot_pos.x), int(robot_pos.y)),
            (int(end.x), int(end.y)),
            width
        )

    if readings:
        chosen_end = robot_pos + pygame.Vector2(
            math.cos(robot_angle + best_offset),
            math.sin(robot_angle + best_offset)
        ) * 26
        pygame.draw.circle(
            screen, YELLOW, (int(chosen_end.x), int(chosen_end.y)), 6
        )


# ============================================================
# 19. PROFESSIONAL ROBOT DRAWING
# ============================================================

def rotated_point(local_x, local_y):
    cos_a = math.cos(robot_angle)
    sin_a = math.sin(robot_angle)

    x = robot_pos.x + local_x * cos_a - local_y * sin_a
    y = robot_pos.y + local_x * sin_a + local_y * cos_a

    return int(x), int(y)


def draw_robot():
    pygame.draw.ellipse(
        screen,
        (6, 9, 14),
        (int(robot_pos.x - 35), int(robot_pos.y + 18), 70, 17)
    )

    for side in (-1, 1):
        wheel_center = rotated_point(0, side * 25)

        wheel_rect = pygame.Rect(
            wheel_center[0] - 8,
            wheel_center[1] - 15,
            16,
            30
        )

        pygame.draw.rect(screen, (7, 9, 13), wheel_rect, border_radius=5)
        pygame.draw.rect(screen, (84, 95, 112), wheel_rect, 2, border_radius=5)

    body = [
        rotated_point(-28, -18),
        rotated_point(15, -18),
        rotated_point(28, 0),
        rotated_point(15, 18),
        rotated_point(-28, 18),
    ]

    pygame.draw.polygon(screen, BLUE, body)
    pygame.draw.polygon(screen, (135, 172, 255), body, 2)

    top = [
        rotated_point(-9, -13),
        rotated_point(14, -13),
        rotated_point(19, 0),
        rotated_point(14, 13),
        rotated_point(-9, 13),
    ]

    pygame.draw.polygon(screen, (33, 46, 65), top)
    pygame.draw.polygon(screen, CYAN, top, 2)

    sensor_center = rotated_point(21, 0)
    pygame.draw.circle(screen, (11, 15, 22), sensor_center, 9)
    pygame.draw.circle(screen, CYAN, sensor_center, 6)

    led_color = PURPLE if open_space_reading < 85 else GREEN
    if escape_mode:
        led_color = RED
    led = rotated_point(-14, 0)
    pygame.draw.circle(screen, led_color, led, 5)


# ============================================================
# 20. SENSOR BEAMS
# ============================================================

def draw_sensor_beams(left, front, right):
    data = [
        ("L", left, robot_angle - math.pi / 4),
        ("F", front, robot_angle),
        ("R", right, robot_angle + math.pi / 4),
    ]

    for label, distance, angle in data:
        end = robot_pos + pygame.Vector2(
            math.cos(angle), math.sin(angle)
        ) * distance

        if distance < 45:
            beam_color = RED
        elif distance < 90:
            beam_color = YELLOW
        else:
            beam_color = CYAN

        pygame.draw.line(
            screen, beam_color,
            (int(robot_pos.x), int(robot_pos.y)),
            (int(end.x), int(end.y)),
            2
        )
        pygame.draw.circle(screen, beam_color, (int(end.x), int(end.y)), 5)

        label_pos = end + pygame.Vector2(6, -12)
        img = TINY.render(f"{label}:{distance:.0f}", True, beam_color)
        screen.blit(img, (int(label_pos.x), int(label_pos.y)))


# ============================================================
# 21. TEXT FUNCTION
# ============================================================

def text(message, x, y, font=FONT, color=WHITE):
    image = font.render(str(message), True, color)
    screen.blit(image, (int(x), int(y)))


# ============================================================
# 22. PANEL
# ============================================================

def panel(rect, border_color=(52, 66, 88)):
    pygame.draw.rect(screen, PANEL, rect, border_radius=14)
    pygame.draw.rect(screen, border_color, rect, 1, border_radius=14)


def section_header(label, x, y, accent=CYAN):
    pygame.draw.rect(screen, accent, pygame.Rect(x, y + 3, 4, 14), border_radius=2)
    text(label, x + 12, y, TINY, MUTED)


# ============================================================
# 23. TOP BAR
# ============================================================

def draw_top_bar():
    pygame.draw.rect(screen, PANEL, TOP_BAR)
    pygame.draw.line(screen, (40, 52, 70), (0, TOP_BAR.bottom), (WIDTH, TOP_BAR.bottom), 2)

    text("AI ROBOT LAB", 32, 16, TITLE, CYAN)
    text(
        "ML-Based Obstacle Avoidance   \u2022   Prepared by Raja Huzaifa Wajid",
        32, 56, SMALL, MUTED
    )

    if finished:
        status, status_color = "MISSION COMPLETE", GREEN
    elif escape_mode:
        status, status_color = "RE-ROUTING", ORANGE
    elif paused:
        status, status_color = "PAUSED", YELLOW
    else:
        status, status_color = "AUTONOMOUS RUNNING", GREEN

    badge = pygame.Rect(WIDTH - 300, 24, 250, 44)
    pygame.draw.rect(screen, PANEL_2, badge, border_radius=22)
    pygame.draw.rect(screen, status_color, badge, 1, border_radius=22)

    pygame.draw.circle(screen, status_color, (badge.x + 22, badge.centery), 7)
    text(status, badge.x + 38, badge.y + 12, SMALL, status_color)

    text("KNN MODEL", WIDTH - 300, 8, TINY, FAINT)


# ============================================================
# 24. SIDEBAR
# ============================================================

def draw_sidebar(left, front, right, action, inference_ms):
    panel(SIDEBAR)

    x = SIDEBAR.x + sc(26)
    y = SIDEBAR.y + sc(24)

    text("ROBOT TELEMETRY", x, y, FONT, WHITE)
    pygame.draw.line(
        screen, (48, 62, 82),
        (x, y + sc(34)), (SIDEBAR.right - sc(26), y + sc(34)), 1
    )

    y += sc(50)

    # --- AI decision card ---
    action_color = {
        "FORWARD": GREEN,
        "LEFT": CYAN,
        "RIGHT": ORANGE,
        "STOP": RED,
        "FINISHED": GREEN,
    }.get(action, WHITE)

    card = pygame.Rect(x - sc(6), y - sc(10), SIDEBAR.width - sc(40), sc(84))
    pygame.draw.rect(screen, PANEL_2, card, border_radius=12)
    pygame.draw.rect(screen, action_color, card, 1, border_radius=12)

    text("AI DECISION", x + sc(10), y, TINY, MUTED)
    text(action, x + sc(10), y + sc(20), BIG, action_color)

    y += sc(98)

    mode_color = ORANGE if escape_mode else (PURPLE if open_space_reading < 85 else GREEN)
    mode_label = "RE-ROUTING" if escape_mode else (
        "SEARCHING GAP" if open_space_reading < 85 else "OPEN PATH"
    )
    text(f"NAV MODE     {mode_label}", x, y, SMALL, mode_color)
    text(f"Open space   {open_space_reading:.0f} cm", x, y + sc(20), TINY, MUTED)

    y += sc(50)

    section_header("ULTRASONIC SENSORS", x, y)
    y += sc(10)

    sensor_values = [("LEFT", left), ("FRONT", front), ("RIGHT", right)]

    for name, value in sensor_values:
        y += sc(26)
        if value < 45:
            c = RED
        elif value < 90:
            c = YELLOW
        else:
            c = CYAN
        text(f"{name:<7} {value:>6.0f} cm", x, y, MONO, c)

    y += sc(36)

    section_header("SYSTEM METRICS", x, y, accent=GOLD)
    y += sc(10)

    elapsed = (
        completion_time
        if finished and completion_time is not None
        else time.perf_counter() - simulation_start
    )

    metrics = [
        f"Dist to goal  {robot_pos.distance_to(TARGET_POS):>7.0f} px",
        f"Collisions    {collision_count:>7}",
        f"AI decisions  {ai_decisions:>7}",
        f"Predictions   {prediction_count:>7}",
        f"Distance      {total_distance:>7.0f} px",
        f"Inference     {inference_ms:>6.2f} ms",
        f"Time          {elapsed:>6.1f} sec",
    ]

    for line in metrics:
        y += sc(23)
        text(line, x, y, MONO_SMALL, WHITE)

    y += sc(32)

    section_header("CONTROLS", x, y, accent=MUTED)
    y += sc(10)

    controls = [
        "SPACE   Pause / Resume",
        "R       Reset mission",
        "ESC     Exit",
    ]

    for line in controls:
        y += sc(22)
        text(line, x, y, SMALL, MUTED)

    # --- credit ---
    pygame.draw.line(
        screen, (48, 62, 82),
        (x, SIDEBAR.bottom - sc(56)), (SIDEBAR.right - sc(26), SIDEBAR.bottom - sc(56)), 1
    )
    text("Prepared by", x, SIDEBAR.bottom - sc(46), TINY, MUTED)
    text("Raja Huzaifa Wajid", x, SIDEBAR.bottom - sc(27), FONT, GOLD)


# ============================================================
# 25. ARENA LABELS
# ============================================================

def draw_arena_labels():
    text(
        "AUTONOMOUS NAVIGATION ARENA",
        ARENA.x + 24, ARENA.y + 16, SMALL, MUTED
    )
    text("START", START.x - 22, START.y + 40, TINY, MUTED)


# ============================================================
# 26. SUCCESS OVERLAY
# ============================================================

def draw_success():
    if not finished:
        return

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((5, 8, 13, 160))
    screen.blit(overlay, (0, 0))

    box = pygame.Rect(WIDTH // 2 - 340, HEIGHT // 2 - 145, 680, 290)

    pygame.draw.rect(screen, PANEL, box, border_radius=20)
    pygame.draw.rect(screen, GREEN, box, 2, border_radius=20)

    text("MISSION COMPLETE", box.centerx - 175, box.y + 40, BIG, GREEN)
    text(
        "Target reached successfully",
        box.centerx - 135, box.y + 96, FONT, WHITE
    )
    text(
        f"Time: {completion_time:.2f}s    "
        f"Collisions: {collision_count}    "
        f"Distance: {total_distance:.0f}px",
        box.centerx - 230, box.y + 142, SMALL, MUTED
    )
    text(
        "Press R to run the mission again",
        box.centerx - 145, box.y + 200, SMALL, CYAN
    )


# ============================================================
# 27. RESET
# ============================================================

def reset():
    global robot_pos
    global robot_angle
    global collision_count
    global ai_decisions
    global prediction_count
    global total_distance
    global last_action
    global path_points
    global simulation_start
    global completion_time
    global finished
    global paused
    global last_sensors
    global smoothed_offset
    global last_scan
    global chosen_offset
    global open_space_reading
    global position_history
    global escape_mode
    global escape_timer
    global escape_turn_dir

    robot_pos = START.copy()
    robot_angle = 0.0

    collision_count = 0
    ai_decisions = 0
    prediction_count = 0

    total_distance = 0.0
    last_action = "FORWARD"

    path_points = []

    simulation_start = time.perf_counter()
    completion_time = None

    finished = False
    paused = False

    last_sensors = (SENSOR_RANGE, SENSOR_RANGE, SENSOR_RANGE)

    smoothed_offset = 0.0
    last_scan = []
    chosen_offset = 0.0
    open_space_reading = SENSOR_RANGE

    position_history = deque(maxlen=POSITION_HISTORY_LEN)
    escape_mode = False
    escape_timer = 0
    escape_turn_dir = random.choice([-1.0, 1.0])


# ============================================================
# 28. START
# ============================================================

print()
print("=" * 70)
print("AI ROBOT LAB - PROFESSIONAL ML SIMULATION (1920x1080)")
print("=" * 70)
print("Model          : KNN")
print("Sensors        : LEFT / FRONT / RIGHT (+ wide radius-aware scan)")
print("Navigation     : Reactive open-space scan, no fixed waypoints")
print("Safety         : Stuck/oscillation watchdog with forced escape")
print("Task           : Autonomous Obstacle Avoidance")
print("Target         : Reach destination safely")
print("Controls       : SPACE = Pause | R = Reset | ESC = Exit")
print("=" * 70)
print()


# ============================================================
# 29. MAIN LOOP
# ============================================================

last_inference_ms = 0.0

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                reset()
            elif event.key == pygame.K_SPACE:
                if not finished:
                    paused = not paused

    if not paused and not finished:
        action, last_inference_ms = update_robot()
        path_points.append(robot_pos.copy())

        if len(path_points) > 6000:
            path_points.pop(0)
    else:
        left, front, right = get_sensors()
        last_sensors = (left, front, right)

    left, front, right = last_sensors

    screen.fill(BACKGROUND)

    draw_top_bar()

    draw_grid()
    draw_arena_labels()
    draw_path()
    draw_obstacles()
    draw_target()
    draw_scan_fan(last_scan, chosen_offset)
    draw_sensor_beams(left, front, right)
    draw_robot()

    draw_sidebar(left, front, right, last_action, last_inference_ms)

    draw_success()

    pygame.display.flip()

    clock.tick(60)


# ============================================================
# 30. FINAL CONSOLE REPORT
# ============================================================

elapsed = (
    completion_time
    if completion_time is not None
    else time.perf_counter() - simulation_start
)

print()
print("=" * 70)
print("FINAL ROBOT SIMULATION REPORT")
print("=" * 70)
print(f"Model            : KNN")
print(f"Target Reached   : {'YES' if finished else 'NO'}")
print(f"Collisions       : {collision_count}")
print(f"AI Decisions     : {ai_decisions}")
print(f"ML Predictions   : {prediction_count}")
print(f"Distance         : {total_distance:.2f} px")
print(f"Completion Time  : {elapsed:.2f} seconds")
print("=" * 70)

pygame.quit()