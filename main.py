import os
from ultralytics import YOLO
import cv2
from team_assigner.team_assigner import TeamAssigner
from visualisation.visualise import visualise_detections

# Paths
CURRENT_DIR = os.getcwd()

# Model path
scratch_model_path = f'{CURRENT_DIR}/models/YOLOV8N_SCRATCH_BEST.pt'
model = YOLO(scratch_model_path)

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

    # Find the player class ID (if not already determined)
    player_class_id = None
    for cls_id, cls_name in model.names.items():
        if cls_name.lower() == "player":
            player_class_id = cls_id
            break

    # Collect player bounding boxes from YOLO results
    player_detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == player_class_id:
                xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                player_detections.append(xyxy)

    # Initialize TeamAssigner
    team_assigner = TeamAssigner()

    # Assign team colours
    team_assigner.assign_team_colour(input_image, player_detections)

    # Visualize results
    output_image = visualise_detections(input_image, results, model, team_assigner, player_class_id, colour_map)

    # Save the output image
    output_path = os.path.join(CURRENT_DIR, "output_single_image.jpg")
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

            # Initialize TeamAssigner
            team_assigner = TeamAssigner()
            
            # input image
            input_image = cv2.imread(image_path)

            # Run YOLO detection
            results = model(image_path)

            # Find the player class ID
            player_class_id = None
            for cls_id, cls_name in model.names.items():
                if cls_name.lower() == "player":
                    player_class_id = cls_id
                    break

            # Collect player bounding boxes from YOLO results
            player_detections = []
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if class_id == player_class_id:
                        xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                        player_detections.append(xyxy)

            # Assign team colours
            team_assigner.assign_team_colour(input_image, player_detections)

            # Visualize results
            output_image = visualise_detections(input_image, results, model, team_assigner, player_class_id, colour_map)

            # Save the output image
            cv2.imwrite(save_path, output_image)
            print(f"Processed and saved: {filename}")

def main():

    # Prompt the user to choose the mode
    mode = input("Enter '1' for a single image test or '2' for a directory of images: ").strip()

    if mode == '1':
        # Prompt for the path to the single image
        test_image_path = 'Football-Players-6/test/images/10_jpg.rf.110b6e7625b4096e4cc1fbfb0f4f43c4.jpg'
        process_single_image(test_image_path)

    elif mode == '2':
        # Prompt for the image directory and save directory
        image_directory = f'{CURRENT_DIR}/Football-Players-6/test/images/'
        save_directory = f'{CURRENT_DIR}/predictions_enhanced_KMeans/'
        process_directory(image_directory, save_directory)

    else:
        print("Invalid input. Please enter either '1' or '2'.")

if __name__ == "__main__":
    main()
