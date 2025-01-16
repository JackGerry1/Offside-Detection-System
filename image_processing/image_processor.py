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
        player_boxes = []
        for r in results:
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                if class_id == self.player_class_id:
                    xyxy = box.xyxy[0].tolist()  # Extract bounding box coordinates as a list
                    player_detections.append(xyxy)
                    player_boxes.append({"coords": xyxy, "id": i})

        # Assign teams
        self.team_assigner = TeamAssigner()
        self.team_assigner.assign_team_colour(input_image, player_detections)

        # Add team and role to each player box
        for player in player_boxes:
            team_id = self.team_assigner.get_player_team(input_image, player['coords'], player_id=player['id'])
            player["team"] = team_id
            player["role"] = None  # Placeholder; roles will be updated during role assignment

        self.player_boxes = player_boxes

        # Visualize initial results
        output_image = visualise_detections(
            input_image, results, self.model, self.team_assigner, self.player_class_id, self.colour_map, "", "", ""
        )

        return output_image

