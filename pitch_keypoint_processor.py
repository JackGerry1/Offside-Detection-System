import os
from ultralytics import YOLO

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the image directories
image_directory_segmentation = f'{CURRENT_DIR}/general_segmentation_images/'
image_directory_keypoints = f'{CURRENT_DIR}/general_keypoints_images/'

# Define the save directories
save_directory_segmentation = f'{CURRENT_DIR}/general_predictions/'
save_directory_keypoints = f'{CURRENT_DIR}/general_keypoints_predictions/'

# Paths to models
segmentation_model_path = f'{CURRENT_DIR}/models/YOLOV8N_BEST.pt'
keypoints_model_path = f'{CURRENT_DIR}/models/YOLOV8NPOSE500BEST.pt'

# Segmentation with pretrained model
model_segmentation = YOLO(segmentation_model_path)

# Create the save directory if it doesn't exist
os.makedirs(save_directory_segmentation, exist_ok=True)

### Test On Multiple Images ###
results_segmentation = model_segmentation(
    source=image_directory_segmentation,
    save=True,
    project=save_directory_segmentation,
    name='.',  # Prevents creation of subfolders
    exist_ok=True  # Allows saving in an existing directory
)

# Keypoints detection
model_keypoints = YOLO(keypoints_model_path)

os.makedirs(save_directory_keypoints, exist_ok=True)

results_keypoints = model_keypoints(
    source=image_directory_keypoints,
    save=True,
    project=save_directory_keypoints,
    name='.',
    exist_ok=True
)