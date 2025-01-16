import cv2
import numpy as np
from utils.utils import draw_bounding_box_with_label

def find_extreme_players(player_boxes, attack_direction):
    """
    Find the furthest forward attacker and furthest back defender based on the attack direction.

    Args:
        player_boxes (list): List of dictionaries with player coordinates and team assignments.
        attack_direction (str): Attack direction, either "left" or "right".

    Returns:
        tuple: Furthest forward attacker, Furthest back defender.
    """
    furthest_forward_attacker = None
    furthest_back_defender = None

    # Determine comparison multipliers based on attack direction
    direction_multiplier = 1 if attack_direction.lower() == "right" else -1

    # Initialize variables for comparison
    extreme_forward_value = -float('inf')
    extreme_back_value = -float('inf')

    for player in player_boxes:
        x_min, _, x_max, _ = player['coords']  # Bounding box coordinates
        role = player['role']  # Attack or Defense

        # Calculate effective comparison value based on direction
        forward_value = direction_multiplier * x_max
        back_value = direction_multiplier * x_min

        # Update furthest forward attacker
        if role == "Attack" and forward_value > extreme_forward_value:
            extreme_forward_value = forward_value
            furthest_forward_attacker = player

        # Update furthest back defender
        if role == "Defense" and back_value > extreme_back_value:
            extreme_back_value = back_value
            furthest_back_defender = player

    return furthest_forward_attacker, furthest_back_defender


def visualise_detections(input_image, results, model, team_assigner, player_class_id, colour_map, team1_role, team2_role, attack_direction, custom_highlights=None):
    """
    Visualize YOLO detection results with class-specific colours, bounding boxes, and masks.

    Args:
        input_image (np.ndarray): The input image.
        results: YOLO detection results.
        model: YOLO model object.
        team_assigner: TeamAssigner instance for assigning team colours.
        player_class_id (int): Class ID for players.
        colour_map (dict): Mapping of class names to BGR colours.
        team1_role (str): Role assigned to Team 1 (e.g., "Attack", "Defense").
        team2_role (str): Role assigned to Team 2 (e.g., "Attack", "Defense").
        attack_direction (str): Direction of the attack ("left" or "right").

    Returns:
        np.ndarray: Image with bounding boxes, masks, and labels visualized.
    """
    output_image = input_image.copy()
    player_boxes = []

    for r in results:
        masks = r.masks.data.cpu().numpy()  # Extract masks as numpy arrays
        for i, box in enumerate(r.boxes):  # Iterate over detections
            xyxy = box.xyxy[0].tolist()
            x_min, y_min, x_max, y_max = map(int, xyxy)
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # Get class name
            confidence = box.conf[0]  # Confidence score

            # Assign colours based on the class name or team
            if class_id == player_class_id:  # Player detection
                team_id = team_assigner.get_player_team(input_image, xyxy, player_id=i)
                role = team1_role if team_id == 1 else team2_role
                colour_bgr = team_assigner.team_colours[team_id]
                player_boxes.append({"coords": xyxy, "team": team_id, "role": role, "index": i})
            else:
                colour_bgr = colour_map[class_name.lower()]  # Colour for non-player objects
            

            # Draw bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), colour_bgr, 2)

            # Create labels
            font_scale = 0.5
            thickness = 1
            text_colour = (255, 255, 255)
            team_label = f"Team {team_id} {role}"
            class_conf_label = f"{class_name} {confidence:.2f}"
            
            label1 = team_label if class_id == player_class_id else ""
            label2 = class_conf_label 
            label_y1 = y_min - 30 if y_min - 30 > 10 else y_min + 20
            label_y2 = label_y1 + 20

            # Add labels to the image
            cv2.putText(output_image, label1, (x_min, label_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour_bgr, thickness * 2)
            cv2.putText(output_image, label2, (x_min, label_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_colour, thickness)
            
            # Apply segmentation mask if available
            mask = masks[i]
            mask = cv2.resize(mask, (input_image.shape[1], input_image.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)
            coloured_mask = np.zeros_like(input_image, dtype=np.uint8)
            coloured_mask[:, :, 0] = colour_bgr[0]
            coloured_mask[:, :, 1] = colour_bgr[1]
            coloured_mask[:, :, 2] = colour_bgr[2]

            output_image = cv2.addWeighted(output_image, 1.0, coloured_mask * mask[:, :, None], 0.5, 0)

    if custom_highlights:
        furthest_forward_attacker = custom_highlights.get("FA")
        furthest_back_defender = custom_highlights.get("FBD")
    else:
        furthest_forward_attacker, furthest_back_defender = find_extreme_players(player_boxes, attack_direction)
    

    # Highlight extreme players
    for extreme_player, colour_label in zip(
    [furthest_forward_attacker, furthest_back_defender], 
    [(0, 255, 0, "FA"), (255, 0, 0, "FBD")]
    ):
        if extreme_player:
            colour, label = colour_label[:3], colour_label[3]
            
            # Print information about the extreme player
            print(f"OG Label: {label} - Player Info: Coords={extreme_player['coords']}, Team={extreme_player['team']}, Role={extreme_player['role']}")
            
            draw_bounding_box_with_label(output_image, extreme_player["coords"], colour, label)

    # Add attack direction indicator
    
    if attack_direction:
        h, _, _ = output_image.shape
        direction_text = f"Attack Direction: {attack_direction.capitalize()}"
        cv2.putText(output_image, direction_text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return output_image