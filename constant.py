# Shared constants for all main scripts
TRAINING_EPISODES = 700


# Window and UI settings
WIDTH = 500
HEIGHT = 680
FPS = 60
WALL_THICKNESS = 15

# Playable Arena Dimensions (Inside the walls)
WALL_WIDTH = 400  # Inner playable width
WALL_HEIGHT = 600  # Inner playable height

# Calculate top-left corner of the playable arena
ARENA_X = (WIDTH - WALL_WIDTH) // 2
ARENA_Y = (HEIGHT - WALL_HEIGHT) // 2

# Agent settings
AGENT_SIZE = 45
START_X = ARENA_X + WALL_WIDTH // 2
START_Y = ARENA_Y + WALL_HEIGHT - 45  # Start safely inside the bottom wall

ANIMATION_MOVE = 0.3
ANIMATION_TURN = 0.3
animation_move = ANIMATION_MOVE
animation_turn = ANIMATION_TURN

# ====================================================
# REWARD VARIABLES (Easily Adjustable)
# ====================================================
MOVE_DIST = 50
TURN_ANGLE = 45
MIN_ANGLE = -90
MAX_ANGLE = 90
MAX_STEPS = 150


# Movement rewards/penalties
STEP_PENALTY = -0  # Base penalty for each step
ASCENT_REWARD = 2  # Reward for moving forward when the center sensor sees no obstacle
TURN_LIMIT_PENALTY = -1.0  # Penalty for trying to turn beyond the allowed -90/+90 range
REPEAT_TURN_PENALTY = -1.0  # Penalty for switching left/right turn actions repeatedly
# Terminal rewards/penalties
TIMEOUT_PENALTY = -20.0  # Penalty for reaching max steps (timeout)
SUCCESS_REWARD = 50.0  # Reward for reaching the finish line
COLLISION_PENALTY = -20.0  # Penalty for any collision (wall or obstacle)

# Movement distances (easily adjustable)
STEP_RANGE = MOVE_DIST / (FPS * ANIMATION_MOVE)  # Pixels per frame
TURNING_RANGE = TURN_ANGLE / (FPS * ANIMATION_TURN)  # Degrees per frame

# Obstacle settings (will be loaded from file if exists)
OBSTACLE_FILE = "obstacle_config.json"
OBS_SIZE = 50
OBS_X = ARENA_X + WALL_WIDTH // 2 - OBS_SIZE // 2
OBS_Y = ARENA_Y + WALL_HEIGHT // 2 - OBS_SIZE // 2
DEFAULT_OBS_SIZE = OBS_SIZE
DEFAULT_OBS_X = OBS_X
DEFAULT_OBS_Y = OBS_Y

# Finish line settings
FINISH_THICKNESS = 5
FINISH_Y = ARENA_Y + 5  # Always positioned near the top

# Sensor settings
MAX_RANGE = 150
SENSOR_ANGLES = [-90, -45, -22.5, 0, 22.5, 45, 90]

# DQN Hyperparameters
STATE_SIZE = 9
ACTION_SIZE = 3
HIDDEN_SIZE = 64
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.001
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.997
TARGET_UPDATE = 10
MEMORY_SIZE = 20000

# Episode settings

# Headless / training defaults
MODEL_SAVE_INTERVAL = 25  # If independent based false only
INDEPENDENT_BASED = True
LOAD_MODEL = "models/best-model.pth"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
DARK_GRAY = (100, 100, 100)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (128, 0, 128)
