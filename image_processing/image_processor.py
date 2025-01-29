# import libraries and functions
import cv2
from ultralytics import YOLO
from team_assigner.team_assigner import TeamAssigner
from visualisation.visualise import visualise_detections, visualise_keypoints

class ImageProcessor:
    # initalise ImageProcessor 
    def __init__(self, model_path, keypoint_modal_path, colour_map):
        # get variabels 
        self.model = YOLO(model_path)
        self.pitch_modal = YOLO(keypoint_modal_path)
        self.colour_map = colour_map
        self.team_assigner = None
        self.processed_results = None
        self.keypoint_results = None
        self.player_class_id = None
        self.input_image = None

    def process_image(self, image_path):
        """
        Processes the image using the trained YOLOV8-seg model and customises the visualisation output. 
        Args:
            image_path: path of the uploaded image.  
        Ultralytics (2024c). Predict - YOLOv8 Docs. [online] Ultralytics. 
        Available at: https://docs.ultralytics.com/modes/predict/ [Accessed 16 Jan. 2025].
        
        Output: Masks and boxes around all recongised classes, with confidence values.  
        """
         
        # Load image
        input_image = cv2.imread(image_path)
        self.input_image = input_image.copy()  # Store a copy for later use

        # Run YOLO detection and save results for future processing. 
        results = self.model(image_path)
        self.processed_results = results 

        # Add path to keypoint model
        keypoint_results = self.pitch_modal(image_path)

        # Find the player class ID, which corresponds to the "player" class
        self.player_class_id = next(
            (cls_id for cls_id, cls_name in self.model.names.items() if cls_name.lower() == "player"), None
        )

        # Collect player bounding boxes and detections. 
        player_detections = []
        player_boxes = []

        # loop through all bounding boxes, inside the results, and appending coords and id of players detected. 
        for r in results:
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                if class_id == self.player_class_id:
                    xyxy = box.xyxy[0].tolist()  
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
            print(f"PLAYER: {player}")

        # store for usage later on. 
        self.player_boxes = player_boxes

        # Visualise initial results, the empty strings represent the team roles and attack direction, which cannot be decided 
        # until the teams are categorised in attack and defense. 
        output_image = visualise_detections(
            input_image, results, self.model, self.team_assigner, self.player_class_id, self.colour_map, "", "", ""
        )

        final_output_image, extracted_keypoints = visualise_keypoints(output_image, keypoint_results)
        self.keypoint_results = extracted_keypoints  # Store keypoints for external access
        
        return final_output_image

