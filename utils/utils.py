import os
import numpy as np

# Define paths and colour map
CURRENT_DIR = os.getcwd()
MODEL_PATH = f'{CURRENT_DIR}/models/YOLOV8N_BEST.pt'
PITCH_MODEL_PATH = f'{CURRENT_DIR}/models/YOLOV8NPOSE500BEST.pt'

# define class ids 
FOOTBALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# define pitch dimensions
PITCH_WIDTH = 69
PITCH_LENGTH = 110

# colour map for bounding boxes of classes 
COLOUR_MAP = {
    "referee": (0, 215, 255),  # Bright Gold
    "football": (255, 255, 255),  # White
    "goalkeeper": (255, 105, 180),  # Pink
}

# Define points corresponding to the new configuration in meters
CONFIG_VERTICES = np.array([
    (0, 0),         # Bottom-left corner of the pitch 
    (0, 14.5),      # Bottom Left Of Box 
    (0, 25.25),     # Bottom Left Of Six Yard Box
    (0, 43.75),     # Top Left Of Six Yard Box
    (0, 54.75),     # Top Left Of Box
    (0, 69),        # Top-left corner of the pitch
    (5.5, 25.25),   # Left Bottom six-yard box outside edge 
    (5.5, 43.75),   # Left Top six-yard box outside edge 
    (11, 34.5),     # Left penalty spot
    (16.5, 14.5),   # Outside Bottom Left box
    (16.5, 27.5),   # Left Penaltiy Arc Bottom
    (16.5, 41),     # Left Penaltiy Arc Top
    (16.5, 54.75),  # Outside Top Left box
    (55, 0),        # In line with bottom of centre circle but at the bottom of the pitch
    (55, 25.25),    # Centre circle bottom edge
    (55, 43.75),    # Centre circle top edge
    (45.75, 34.5),  # Centre circle left side
    (64, 34.5),     # Centre circle right side
    (55, 69),       # In line with top of centre circle but at the top of the pitch
    (93.5, 14.5),   # Outside Bottom Right box
    (93.5, 28),     # Right Penaltiy Arc Bottom
    (93.5, 41),     # Right Penaltiy Arc Top
    (93.5, 54.75),  # Outside Top Right box
    (99, 34.5),     # Right penalty spot
    (110, 0),       # Bottom-right corner of the pitch
    (110, 14.5),    # Bottom Right box
    (110, 25.25),   # Bottom Right Of Six Yard Box
    (110, 43.75),   # Top Right Of Six Yard Box
    (104.5, 25.25), # Right Bottom six-yard box outside edge
    (104.5, 43.75), # Right Top six-yard box outside edge
    (110, 54.75),   # Top Right Of Box
    (110, 69),      # Top-right corner of the pitch
])