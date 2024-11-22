import os 
from ultralytics import YOLO

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the image and save directories
image_directory = f'{CURRENT_DIR}/Football-Players-6/test/images/'
save_directory_scratch = f'{CURRENT_DIR}/predictions_scratch/'
save_directory_pretrained = f'{CURRENT_DIR}/predictions_pretrained/'

# paths to models 
pretrained_model_path = f'{CURRENT_DIR}/models/YOLOV8N_BEST_PRETRAINED.pt'
scratch_model_path = f'{CURRENT_DIR}/models/YOLOV8N_SCRATCH_BEST.pt'

# load pretrained model
model = YOLO(pretrained_model_path)

# Create the save directory if it doesn't exist
os.makedirs(save_directory_pretrained, exist_ok=True)

### Test On Multiple Images ###
results = model(
    source=image_directory, 
    save=True, 
    project=save_directory_pretrained,
    name='.',        # Prevents creation of subfolders
    exist_ok=True    # Allows saving in an existing directory
)
