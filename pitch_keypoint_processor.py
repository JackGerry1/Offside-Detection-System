import os 
from ultralytics import YOLO

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the image and save directories
image_directory = f'{CURRENT_DIR}/pitch_keypoints_potential_offsides/'
save_directory = f'{CURRENT_DIR}/500_epoch_pitch_keypoints_test/'

# paths to models 
modal_path = f'{CURRENT_DIR}/models/YOLOV8NPOSE500BEST.pt'

# load pretrained model
model = YOLO(modal_path)

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

### Test On Multiple Images ###
results = model(
    source=image_directory, 
    save=True, 
    project=save_directory,
    name='.',        # Prevents creation of subfolders
    exist_ok=True    # Allows saving in an existing directory
)