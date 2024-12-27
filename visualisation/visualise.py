import cv2
import numpy as np

def visualise_detections(input_image, results, model, team_assigner, player_class_id, colour_map):
    """
    Visualise YOLO detection results with class-specific colours, confidence scores, and masks.

    Args:
        input_image (np.ndarray): The input image.
        results: YOLO detection results.
        model: YOLO model object.
        team_assigner: TeamAssigner instance for assigning team colours.
        player_class_id (int): Class ID for players.
        colour_map (dict): Mapping of class names to BGR colours.

    Returns:
        np.ndarray: Image with bounding boxes, masks, and labels visualized.
    """
    output_image = input_image.copy()
    
    for r in results:
        masks = r.masks.data.cpu().numpy()  # Convert masks to numpy arrays
        for i, box in enumerate(r.boxes):  # Iterate over the bounding boxes
            # Extract bounding box coordinates
            xyxy = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, xyxy)

            # Extract confidence and class details
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = model.names[class_id]  # Class name

            # Assign colours based on the class name
            if class_name.lower() in colour_map:
                colour_bgr = colour_map[class_name.lower()]  # Use the defined colour map
            elif class_id == player_class_id:  
                # For players, use team-specific colours
                team_id = team_assigner.get_player_team(input_image, xyxy, player_id=i)
                team_colour = team_assigner.team_colours[team_id]
                colour_bgr = tuple(map(int, team_colour))  # Convert to BGR

            # Draw bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), colour_bgr, 2)

            # Create and position labels
            font_scale = 0.5
            thickness = 1
            text_colour = (255, 255, 255)
            label1 = f"Team {team_id}" if class_name.lower() == "player" else ""
            label2 = f"{class_name} {confidence:.2f}"
            label_y1 = y_min - 30 if y_min - 30 > 10 else y_min + 20
            label_y2 = label_y1 + 20

            # Add labels to the image
            cv2.putText(output_image, label1, (x_min, label_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour_bgr, thickness)
            cv2.putText(output_image, label2, (x_min, label_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, thickness)

            # Apply segmentation masks if available
            mask = masks[i]
            mask = cv2.resize(mask, (input_image.shape[1], input_image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            coloured_mask = np.zeros_like(input_image, dtype=np.uint8)  # Create an empty coloured mask
            coloured_mask[:, :, 0] = colour_bgr[0]  # Assign blue channel
            coloured_mask[:, :, 1] = colour_bgr[1]  # Assign green channel
            coloured_mask[:, :, 2] = colour_bgr[2]  # Assign red channel

            # Add the mask to the respective output image
            output_image = cv2.addWeighted(output_image, 1.0, coloured_mask * mask[:, :, None], 0.5, 0)
    return output_image
