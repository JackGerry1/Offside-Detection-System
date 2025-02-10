import os
from ultralytics import YOLO

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the image directories
image_directory_keypoints = f'{CURRENT_DIR}/football-field-detection-v1/test/images/'
image_directory_players = f'{CURRENT_DIR}/football-players-detection-1/test/images/'

# Define the save directories
save_directory_keypoints = f'{CURRENT_DIR}/keypoint_predictions/'
save_directory = f'{CURRENT_DIR}/player_referee_football_goalkeeper_predictions/'

# Paths to models
model_path = f'{CURRENT_DIR}/models/YOLOV8N_BEST.pt'
keypoints_model_path = f'{CURRENT_DIR}/models/YOLOV8NPOSE500BEST.pt'

# Player, Referee, Football and Goalkeeper detection with pretrained model
modal = YOLO(model_path)

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

### Test On Multiple Images ###
results_segmentation = modal(
    source=image_directory_players,
    save=True,
    project=save_directory,
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