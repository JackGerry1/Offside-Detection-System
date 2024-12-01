import os 
from ultralytics import YOLO
import cv2
import numpy as np

# Get the current working directory
CURRENT_DIR = os.getcwd()

# Define the image and save directories
image_directory = f'{CURRENT_DIR}/Football-Players-6/test/images/'
save_directory_scratch = f'{CURRENT_DIR}/predictions_scratch/'
save_directory_pretrained = f'{CURRENT_DIR}/predictions_pretrained/'
test_directory_scratch = f'{CURRENT_DIR}/test_prediction/'
test_image_path = 'Football-Players-6/train/images/37_jpg.rf.9d8350fda1247ff3b266ae2d577e024a.jpg'
# Paths to models 
pretrained_model_path = f'{CURRENT_DIR}/models/YOLOV8N_BEST_PRETRAINED.pt'
scratch_model_path = f'{CURRENT_DIR}/models/YOLOV8N_SCRATCH_BEST.pt'

# Load pretrained model
model = YOLO(pretrained_model_path)

# Create the save directory if it doesn't exist
os.makedirs(test_directory_scratch, exist_ok=True)

### Test On Single Image
results = model(
    source=test_image_path, 
    save=True, 
    project=test_directory_scratch, 
    name='.',        
    exist_ok=True    
)
# Load the original YOLO-styled output image
#output_image_path = os.path.join(test_directory_scratch, "37_jpg.rf.9d8350fda1247ff3b266ae2d577e024a.jpg")
image = cv2.imread("Football-Players-6/train/images/37_jpg.rf.9d8350fda1247ff3b266ae2d577e024a.jpg")

# Define a color palette (example with 10 colors, you can expand it as needed)
color_palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128),
    (0, 128, 128), (128, 128, 128)
]

# Process results
for r in results:
    if r.masks is not None:  # Check if masks are available
        masks = r.masks.data.cpu().numpy()  # Convert masks to numpy arrays

        # Iterate over each detected object
        for i, box in enumerate(r.boxes):  
            # Extract data
            xyxy = box.xyxy[0].tolist()  # Convert to list
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name
            
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, xyxy)  # Convert to integers
            
            # Get color for the class
            color = color_palette[class_id % len(color_palette)]  # Cycle through palette if classes exceed colors

            # Apply mask with transparency
            mask = masks[i]
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to match image size
            mask = (mask > 0.5).astype(np.uint8)  # Binarize mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            colored_mask[:, :, 0] = color[0]
            colored_mask[:, :, 1] = color[1]
            colored_mask[:, :, 2] = color[2]

            # Blend mask with the original image
            image = cv2.addWeighted(image, 1.0, colored_mask * mask[:, :, None], 0.5, 0)

            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # Format and add text for class, confidence, and xyxy positions
            label1 = f"{class_name} {confidence:.2f}"
            label2 = f"X1, Y1: ({x_min}, {y_min})"
            label3 = f"X2, Y2: ({x_max}, {y_max})"

            # Text positions
            text_y1 = y_min - 50 if y_min - 50 > 10 else y_min + 18
            text_y2 = text_y1 + 14  # Adjusted for smaller font
            text_y3 = text_y2 + 14  # Adjusted for smaller font

            # Add text with smaller font and class color
            font_scale = 0.35  # Smaller font scale
            thickness = 1  # Optional: Adjust text thickness
            cv2.putText(image, label1, (x_min, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(image, label2, (x_min, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(image, label3, (x_min, text_y3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Save the updated image with masks, bounding boxes, and annotations
output_path = os.path.join(test_directory_scratch, "output_with_masks_and_text.jpg")
cv2.imwrite(output_path, image)
print(f"Output image saved at: {output_path}")
