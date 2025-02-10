import cv2
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

PITCH_WIDTH = 69
PITCH_LENGTH = 110

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

def draw_bounding_box_with_label(image, coords, colour, label, font_scale=0.6, thickness=4, text_colour=(255, 255, 255)):
    """
    Draws a bounding box with a label on the image.
    
    Args:
        image (np.ndarray): The image on which to draw.
        coords (tuple): Coordinates of the bounding box (x_min, y_min, x_max, y_max).
        colour (tuple): BGR colour for the bounding box.
        label (str): Text label to display.
        font_scale (float): Font scale for the text.
        thickness (int): Thickness of the bounding box and text.
        text_colour (tuple): BGR colour for the text.
    
    References: 
        Gallagher, J. (2023). How to Draw a Bounding Box Prediction Label with Python. [online] Roboflow Blog. 
        Available at: https://blog.roboflow.com/how-to-draw-a-bounding-box-label-python/ [Accessed 11 Dec. 2024].
        
        OpenCV (2024). OpenCV: Drawing Functions in OpenCV. [online] Opencv.org. 
        Available at: https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html [Accessed 11 Dec. 2024].

    Output: 
        The bounding boxes for with the labels for furthest forward attacker and furthest back defender.  
    """
    x_min, y_min, x_max, y_max = map(int, coords)
    
    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour, thickness)
    
    # Add label below the bounding box
    label_x = x_min
    label_y = y_max + 20  # Slightly below the box
    
    # Add a filled background for the text
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    label_background = (
        (label_x, label_y - label_size[1] - 5),
        (label_x + label_size[0] + 5, label_y + 5)
    )
    
    # draw the bounding box and add the labels text. 
    cv2.rectangle(image, label_background[0], label_background[1], colour, -1)
    cv2.putText(
        image, label, (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, int(thickness * 0.5)
    )
