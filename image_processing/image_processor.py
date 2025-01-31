# import libraries and functions
import cv2
from ultralytics import YOLO
from team_assigner.team_assigner import TeamAssigner
from visualisation.visualise import visualise_detections, visualise_keypoints
from utils.utils import FOOTBALL_CLASS_ID, GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID

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
        self.goalkeeper_results = None
        self.referee_results = None
        self.football_results = None
        self.player_boxes = None
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

        player_detections = []
        player_boxes = []
        referee_boxes = []
        football_boxes = []
        goalkeeper_boxes = []

        # Dictionary to map class IDs to their respective lists
        class_map = {
            PLAYER_CLASS_ID: (player_detections, player_boxes),
            REFEREE_CLASS_ID: (referee_boxes, None),
            FOOTBALL_CLASS_ID: (football_boxes, None),
            GOALKEEPER_CLASS_ID: (goalkeeper_boxes, None),
        }
        # Collect player bounding boxes and detections. 
        
        # loop through all bounding boxes, inside the results, and appending coords and id of players detected. 
        for r in results:
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
        
                if class_id in class_map:
                    target_list, player_box_list = class_map[class_id]
                    target_list.append(xyxy)
        
                    # Only add to player_boxes if it's a player detection
                    if player_box_list is not None:
                        player_box_list.append({"coords": xyxy, "id": i})
        # Assign teams
        self.team_assigner = TeamAssigner()
        self.team_assigner.assign_team_colour(input_image, player_detections)

        # Add team and role to each player box
        for player in player_boxes:
            team_id = self.team_assigner.get_player_team(input_image, player['coords'], player_id=player['id'])
            player["team"] = team_id
            player["role"] = None  # Placeholder; roles will be updated during role assignment

        # store for usage later on. 
        self.player_boxes = player_boxes
        self.goalkeeper_results = goalkeeper_boxes
        self.referee_results = referee_boxes
        self.football_results = football_boxes

        # Visualise initial results, the empty strings represent the team roles and attack direction, which cannot be decided 
        # until the teams are categorised in attack and defense. 
        output_image = visualise_detections(
            input_image, results, self.model, self.team_assigner, PLAYER_CLASS_ID, self.colour_map, "", "", ""
        )

        final_output_image, extracted_keypoints = visualise_keypoints(output_image, keypoint_results)
        self.keypoint_results = extracted_keypoints  # Store keypoints for external access
        
        return final_output_image

