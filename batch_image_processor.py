import os
from ultralytics import YOLO
import cv2
from team_assigner.team_assigner import TeamAssigner
from visualisation.visualise import visualise_detections
from utils.utils import PLAYER_CLASS_ID, MODEL_PATH

# Paths
CURRENT_DIR = os.getcwd()

# Model path
model = YOLO(MODEL_PATH)

# Define colours for different classes
colour_map = {
    "referee": (0, 0, 0),  # Black
    "football": (0, 165, 255),  # Orange
    "goalkeeper": (255, 105, 180),  # Pink
}

def process_single_image(image_path):
    input_image = cv2.imread(image_path)
    
    # Run YOLO detection
    results = model(image_path)


    # Collect player bounding boxes from YOLO results
    player_detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == PLAYER_CLASS_ID:
                xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                player_detections.append(xyxy)

    # Process the detected bounding boxes
    for r in results:
        for i, box in enumerate(r.boxes):  # Access bounding boxes
            class_id = int(box.cls[0])  # Class ID for the detection
            if class_id == PLAYER_CLASS_ID:  # Only process "player" detections
                xyxy = box.xyxy[0].tolist()  # Get bounding box coordinates [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = map(int, xyxy)  # Convert to integers

                # Crop the bounding box from the original image
                cropped_image = input_image[y_min:y_max, x_min:x_max]
                # Save the cropped image
                cropped_image_path = os.path.join(CURRENT_DIR, f"cropped_player_{i}.jpg")
                cv2.imwrite(cropped_image_path, cropped_image)

                print(f"Cropped player image saved at: {cropped_image_path}")

    # Initialise TeamAssigner
    team_assigner = TeamAssigner()

    # Assign team colours
    team_assigner.assign_team_colour(input_image, player_detections)

    # Visualise results
    output_image, _ = visualise_detections(input_image, results, model, team_assigner, PLAYER_CLASS_ID, colour_map, "", "", "")

    # Save the output image
    output_path = os.path.join(CURRENT_DIR, "output_test_image.jpg")

    cv2.imwrite(output_path, output_image)
    print(f"Output image saved at: {output_path}")

def process_directory(image_directory, save_directory):
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
            output_image, _ = visualise_detections(input_image, results, model, team_assigner, PLAYER_CLASS_ID, colour_map, "", "", "")

            # Save the output image
            cv2.imwrite(save_path, output_image)
            print(f"Processed and saved: {filename}")

def main():

    # Prompt the user to choose the mode
    mode = input("Enter '1' for a single image test or '2' for a directory of images: ").strip()

    if mode == '1':
        # Prompt for the path to the single image
        test_image_path = 'football-field-detection-v1/test/images/121364_3_1_png.rf.dd1f0870fd78f5a2f6fe697784d97d2c.jpg'
        process_single_image(test_image_path)

    elif mode == '2':
        # Prompt for the image directory and save directory
        image_directory = f'{CURRENT_DIR}/football-field-detection-v1/test/images'
        save_directory = f'{CURRENT_DIR}/team_colour_predictions3/'
        process_directory(image_directory, save_directory)

    else:
        print("Invalid input. Please enter either '1' or '2'.")

if __name__ == "__main__":
    main()
