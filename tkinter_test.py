import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
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

# Global variable to store the uploaded image path
uploaded_image_path = None

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

    # Initialise TeamAssigner
    team_assigner = TeamAssigner()

    # Assign team colours
    team_assigner.assign_team_colour(input_image, player_detections)

    # Visualise results
    output_image = visualise_detections(input_image, results, model, team_assigner, player_class_id, colour_map)

    # Save the output image
    output_path = os.path.join(CURRENT_DIR, "output_single_image.jpg")
    cv2.imwrite(output_path, output_image)
    return output_path

def upload_and_display_image():
    global uploaded_image_path
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    
    if file_path:
        uploaded_image_path = file_path  # Save the uploaded image path
        # Load the image
        img = Image.open(file_path)
        img_tk = ImageTk.PhotoImage(img)
        
        # Display the image in the label
        image_label.config(image=img_tk)
        image_label.image = img_tk

def process_and_display_image():
    global uploaded_image_path
    if uploaded_image_path:
        # Process the image and get the path of the resulting image
        result_image_path = process_single_image(uploaded_image_path)
        
        # Load the resulting image
        result_img = Image.open(result_image_path)
        result_img_tk = ImageTk.PhotoImage(result_img)
        
        # Display the resulting image in the label
        image_label.config(image=result_img_tk)
        image_label.image = result_img_tk
    else:
        print("No image uploaded!")

# Create the main application window
window = tk.Tk()
window.title("Image Upload and Processing GUI")

# Set the window size to 800x800
window.geometry("800x800")

# Add a button to upload the image
upload_button = tk.Button(window, text="Upload Image", command=upload_and_display_image)
upload_button.pack(pady=10)

# Add a button to process the uploaded image
process_button = tk.Button(window, text="Process Image", command=process_and_display_image)
process_button.pack(pady=10)

# Add a label to display the uploaded and processed image
image_label = tk.Label(window)
image_label.pack()

# Run the application
window.mainloop()
