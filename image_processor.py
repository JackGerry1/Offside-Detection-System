import os
import cv2
from ultralytics import YOLO
from team_assigner.team_assigner import TeamAssigner
from visualisation.visualise import visualise_detections

class ImageProcessor:
    def __init__(self, model_path, colour_map):
        self.model = YOLO(model_path)
        self.colour_map = colour_map
        self.team_assigner = None
        self.processed_results = None
        self.player_class_id = None
        self.input_image = None

    def process_image(self, image_path):
        # Load image
        input_image = cv2.imread(image_path)
        self.input_image = input_image.copy()  # Store a copy for later use

        # Run YOLO detection
        results = self.model(image_path)
        self.processed_results = results  # Save the results

        # Find the player class ID
        self.player_class_id = next(
            (cls_id for cls_id, cls_name in self.model.names.items() if cls_name.lower() == "player"), None
        )

        # Collect player bounding boxes
        player_detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id == self.player_class_id:
                    xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                    player_detections.append(xyxy)

        # Assign teams
        self.team_assigner = TeamAssigner()
        self.team_assigner.assign_team_colour(input_image, player_detections)

        # Visualise initial results
        output_image = visualise_detections(
            input_image, results, self.model, self.team_assigner, self.player_class_id, self.colour_map, "", "", ""
        )

        return output_image

    def update_roles(self, team1_role, team2_role, attack_direction):
        if self.processed_results and self.team_assigner:
            updated_image = visualise_detections(
                self.input_image, self.processed_results, self.model, self.team_assigner,
                self.player_class_id, self.colour_map, team1_role, team2_role, attack_direction
            )
            return updated_image
        raise ValueError("No processed results available.")