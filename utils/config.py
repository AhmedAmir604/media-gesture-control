"""
Configuration file for Gesture Media Control System
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GESTURES_DIR = os.path.join(DATA_DIR, "gestures")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GESTURES_DIR, exist_ok=True)

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# MediaPipe settings
MEDIAPIPE_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# Gesture recognition settings
GESTURE_CLASSES = [
    'open_palm',     # Play/Pause
    'fist',          # Stop
    'thumbs_up',     # Volume Up
    'thumbs_down',   # Volume Down
    'peace',         # Next Track
    'ok_sign',       # Previous Track
    'pointing',      # Mute/Unmute
    'none'           # No gesture
]

# Media control mappings
GESTURE_COMMANDS = {
    'open_palm': 'playpause',
    'fist': 'stop',
    'thumbs_up': 'volume_up',
    'thumbs_down': 'volume_down',
    'peace': 'next_track',
    'ok_sign': 'prev_track',
    'pointing': 'mute',
    'none': None
}

# Model settings
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_classifier.h5")
INPUT_SIZE = 63  # 21 landmarks * 3 coordinates (x, y, z)
HIDDEN_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32

# Data collection settings
SAMPLES_PER_GESTURE = 1000
COLLECTION_DELAY = 0.1  # seconds between samples

# GUI settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
PREVIEW_WIDTH = 400
PREVIEW_HEIGHT = 300

# Performance settings
GESTURE_STABILITY_FRAMES = 5  # Number of consistent frames needed
COMMAND_COOLDOWN = 1.0  # Seconds between same command execution
CONFIDENCE_THRESHOLD = 0.8

# Colors (BGR format for OpenCV)
COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}

# Landmarks indices for different hand parts
HAND_LANDMARKS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
    'palm': [0, 1, 5, 9, 13, 17]
} 