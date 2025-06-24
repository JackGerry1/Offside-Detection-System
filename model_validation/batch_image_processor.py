import os
import sys
from ultralytics import YOLO
import cv2

# Add the parent directory to Python path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from team_assigner.team_assigner import TeamAssigner
from visualisation.visualise import visualise_detections, visualise_keypoints
from utils.utils import PLAYER_CLASS_ID, MODEL_PATH, COLOUR_MAP, PITCH_MODEL_PATH

CURRENT_DIR = os.getcwd()

# Model paths
model = YOLO(MODEL_PATH)
keypoint_modal = YOLO(PITCH_MODEL_PATH)

def process_single_image(image_path):
    """
    Process single image with YOLO modal. 

    Args:
        image_path: single input image
    Outputs:
        image with bounding boxes for detected classes. 
    """
    input_image = cv2.imread(image_path)

    # Run YOLO detection
    results = model(image_path)
    keypoint_results = keypoint_modal(image_path)

    # Collect player bounding boxes from YOLO results
    player_detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == PLAYER_CLASS_ID:
                xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                player_detections.append(xyxy)

    # Initialise TeamAssigner
    team_assigner = TeamAssigner()

    # Assign team colours
    team_assigner.assign_team_colour(input_image, player_detections)

    # Visualise results
    output_image, _ = visualise_detections(input_image, results, model, team_assigner, PLAYER_CLASS_ID, COLOUR_MAP, "", "", "")
    
    final_output_image, _ = visualise_keypoints(output_image, keypoint_results)

    # Save the output image
    output_path = os.path.join(CURRENT_DIR, "output_test_image.jpg")

    cv2.imwrite(output_path, final_output_image)
    print(f"Output image saved at: {output_path}")

def process_directory(image_directory, save_directory):
    """
    Process with YOLO for all images in a directory. 

    Args:
        image_directory: image directory that is to be processed. 
        save_directory: where the output images are saved. 
    Outputs:
        Processed directory of images. 
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Process all images in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
            image_path = os.path.join(image_directory, filename)
            save_path = os.path.join(save_directory, filename)

            # input image
            input_image = cv2.imread(image_path)

            # Run YOLO detection
            results = model(image_path)
            keypoint_results = keypoint_modal(image_path)
            
            # Collect player bounding boxes from YOLO results
            player_detections = []
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if class_id == PLAYER_CLASS_ID:
                        xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                        player_detections.append(xyxy)

            # Initialise TeamAssigner
            team_assigner = TeamAssigner()

            # Assign team colours
            team_assigner.assign_team_colour(input_image, player_detections)

            # Visualise results
            output_image, _ = visualise_detections(input_image, results, model, team_assigner, PLAYER_CLASS_ID, COLOUR_MAP, "", "", "")
            
            final_output_image, _ = visualise_keypoints(output_image, keypoint_results)

            # Save the output image
            save_path = os.path.join(CURRENT_DIR, "output_test_image.jpg")

            # Save the output image
            cv2.imwrite(save_path, final_output_image)
            print(f"Processed and saved: {filename}")

def main():
    """
    Chooses whether single image or image directory is processed.  

    Outputs:
        Images with bounding boxes, masks, and labels visualised.
    """
    # Prompt the user to choose the mode
    mode = input("Enter '1' for a single image test or '2' for a directory of images: ").strip()

    if mode == '1':
        # Prompt for the path to the single image
        test_image_path = 'datasets/football-field-detection-v1/test/images/121364_3_1_png.rf.dd1f0870fd78f5a2f6fe697784d97d2c.jpg'
        process_single_image(test_image_path)

    elif mode == '2':
        # Prompt for the image directory and save directory
        image_directory = f'{CURRENT_DIR}/datasets/football-field-detection-v1/test/images'
        save_directory = f'{CURRENT_DIR}/team_colour_predictions2/'
        process_directory(image_directory, save_directory)

    else:
        print("Invalid input. Please enter either '1' or '2'.")

if __name__ == "__main__":
    main()
