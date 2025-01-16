import cv2
import os

# Define paths and colour map
CURRENT_DIR = os.getcwd()
MODEL_PATH = f'{CURRENT_DIR}/models/YOLOV8N_SCRATCH_BEST.pt'
COLOUR_MAP = {
    "referee": (0, 0, 0),  # Black
    "football": (0, 165, 255),  # Orange
    "goalkeeper": (255, 105, 180),  # Pink
}


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
    cv2.rectangle(image, label_background[0], label_background[1], colour, -1)
    cv2.putText(
        image, label, (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, int(thickness * 0.5)
    )
