# import libraries 
import cv2

def visualise_detections(input_image, results, model, team_assigner, player_class_id, colour_map, team1_role, team2_role, attack_direction):
    """
    Visualise YOLO detection results with class-specific colours, bounding boxes, and masks.

    Args:
        input_image (np.ndarray): The input image.
        results: YOLO detection results.
        model: YOLO model object.
        team_assigner: TeamAssigner instance for assigning team colours.
        player_class_id (int): Class ID for players.
        colour_map (dict): Mapping of class names to BGR colours.
        team1_role (str): Role assigned to Team 1 ("Attack" or "Defence").
        team2_role (str): Role assigned to Team 2 ("Attack" or "Defence").
        attack_direction (str): Direction of the attack ("left" or "right").
    
    References: 
        Gallagher, J. (2023). How to Draw a Bounding Box Prediction Label with Python. [online] Roboflow Blog. 
        Available at: https://blog.roboflow.com/how-to-draw-a-bounding-box-label-python/ [Accessed 11 Dec. 2024].
        
        OpenCV (2024). OpenCV: Drawing Functions in OpenCV. [online] Opencv.org. 
        Available at: https://docs.opencv.org/3.4/dc/da5/tutorial_py_drawing_functions.html [Accessed 11 Dec. 2024].

        Ultralytics (2024c). Predict - YOLOv8 Docs. [online] Ultralytics. 
        Available at: https://docs.ultralytics.com/modes/predict/ [Accessed 1 Jan. 2025].

    Outputs:
        Image with bounding boxes, masks, and labels visualised.
    """

    # initialise output image and empty player_boxes array. 
    output_image = input_image.copy()
    player_boxes = []
    team_id = None
    role = None
    
    # for all results, extract information about boxes, masks and classes. 
    for r in results:
        for i, box in enumerate(r.boxes): 

            # extract coordinates, class name and confidence values for all detections.  
            xyxy = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, xyxy)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  
            confidence = box.conf[0] 

            # Assign Players bounding box and mask colour based on team_assigner results
            # In addition to roles, and appending this information to player boxes for later.  
            if class_id == player_class_id:  
                team_id = team_assigner.get_player_team(input_image, xyxy, player_id=i)
                role = team1_role if team_id == 1 else team2_role
                colour_bgr = team_assigner.team_colours[team_id]
                player_boxes.append({"coords": xyxy, "team": team_id, "role": role, "index": i, "team_colour": colour_bgr})

            # assign other classes colours based on colour map.  
            else:
                colour_bgr = colour_map[class_name.lower()] 
            
            # Draw bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), colour_bgr, 2)

            # Create labels
            font_scale = 0.5
            thickness = 1
            text_colour = (255, 255, 255)
            team_label = f"Team {team_id} {role}"
            class_conf_label = f"{class_name} {confidence:.2f}"
            
            # position labels nicely
            label1 = team_label if class_id == player_class_id else ""
            label2 = class_conf_label 
            label_y1 = y_min - 30 if y_min - 30 > 10 else y_min + 20
            label_y2 = label_y1 + 20

            # Add labels to the image
            cv2.putText(output_image, label1, (x_min, label_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour_bgr, thickness * 2)
            cv2.putText(output_image, label2, (x_min, label_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, thickness)
            
    # Add attack direction indicator
    if attack_direction:
        h, _, _ = output_image.shape
        direction_text = f"Attack Direction: {attack_direction.capitalize()}"
        cv2.putText(output_image, direction_text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return output_image, player_boxes

def visualise_keypoints(saved_image, keypoint_results):
    """
    Visualise keypoints if over 0.5 and store the rest. 

    Args:
        saved_image: image uploaded by the user. 
        keypoint_results: The keypoint_results detected from the image. 
    Outputs:
        Image with bounding boxes, masks, and labels visualised.
    """
    output_image = saved_image.copy()
    keypoints_data = [] 
    
    # Add keypoint rendering
    if keypoint_results:
        # loop through all keypoints
        for r in keypoint_results:
            keypoints = r.keypoints.data.cpu().numpy()
            for kp_set in keypoints:
                for kp in kp_set:
                    # extract x y and confidence for each keypoints
                    x, y, conf = kp

                    # append keypoints_data for homography transformation based on 2D pitch keypoints 
                    # all keypoints are appended so the matching 2D pitch keypoints can be filtered. 
                    keypoints_data.append(kp)
                    
                    # Only render keypoints with confidence > 0.5
                    if conf > 0.5:  
                        cv2.circle(output_image, (int(x), int(y)), 5, (0, 255, 255), -1)
    
    return output_image, keypoints_data