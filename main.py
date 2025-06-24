# import neccessary libraries
import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
import os
import cv2
from visualisation.pitch_visualiser import pitch_display
from pitch_transformer.position_transformer import PositionTransformer
from pitch_transformer.coordinate_transformer import CoordinateTransformer
from image_processing.image_processor import ImageProcessor
from visualisation.visualise import visualise_detections
from utils.utils import CURRENT_DIR, MODEL_PATH, PITCH_MODEL_PATH, COLOUR_MAP, PLAYER_CLASS_ID

# Initialise ImageProcessor
processor = ImageProcessor(MODEL_PATH, PITCH_MODEL_PATH, COLOUR_MAP)
coordinate_transformer = CoordinateTransformer()
position_transformer = PositionTransformer()

class ImageApp:
    # GUI initialisation
    def __init__(self, root):
        # title and dimensions setup
        self.root = root
        self.root.title("Offside Detection Prototype")
        self.root.geometry("800x950")

        # initialsing paths for upload image result image custom highlights and processed players
        self.uploaded_image_path = None
        self.result_image_path = None
        self.custom_highlights = {"FA": None, "FBD": None}
        self.processed_players = []

        self.init_components()

    def init_components(self):
        """
        initialises the components for the GUI, such as buttons labels and dropdown selections. 

        Output: 
            GUI with the relevant buttons, labels and dropdown selections, with styling and positioning  
        """
        
        # Upload button
        upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_and_display_image)
        upload_button.pack(pady=10)

        # Process button
        process_button = tk.Button(self.root, text="Process Image", command=self.process_and_display_image)
        process_button.pack(pady=10)

        # Image label with mouse1 onclick functionality
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        
        # Role assignment drop down menu for team role and attack direction.
        self.assign_roles_frame = tk.Frame(self.root)
        self.team1_role_var = StringVar(value="Attack")
        self.team2_role_var = StringVar(value="Defence")
        self.attack_direction_var = StringVar(value="Left")  

        # Team role selection
        team1_label = tk.Label(self.assign_roles_frame, text="Team 1 Role:")
        team1_label.grid(row=0, column=0, padx=5, pady=5)
        team1_role_menu = tk.OptionMenu(self.assign_roles_frame, self.team1_role_var, "Attack", "Defence")
        team1_role_menu.grid(row=0, column=1, padx=5, pady=5)

        team2_label = tk.Label(self.assign_roles_frame, text="Team 2 Role:")
        team2_label.grid(row=1, column=0, padx=5, pady=5)
        team2_role_menu = tk.OptionMenu(self.assign_roles_frame, self.team2_role_var, "Attack", "Defence")
        team2_role_menu.grid(row=1, column=1, padx=5, pady=5)

        # Attack direction selection
        direction_label = tk.Label(self.assign_roles_frame, text="Attack Direction:")
        direction_label.grid(row=2, column=0, padx=5, pady=5)
        direction_menu = tk.OptionMenu(self.assign_roles_frame, self.attack_direction_var, "Left", "Right")
        direction_menu.grid(row=2, column=1, padx=5, pady=5)

        # Assign roles button
        assign_button = tk.Button(self.assign_roles_frame, text="Assign Roles", command=self.assign_roles)
        assign_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.assign_roles_frame.pack_forget()


        # Create Offside Detection button but hide it initially
        self.offside_detection_button= tk.Button(self.root, text="Detect Offside", command=self.pass_data)
        self.offside_detection_button.pack()
        self.offside_detection_button.pack_forget() 


    def upload_and_display_image(self):
        """
        Uploads user's image and displays it on the GUI 

        Output: 
            The image uploaded by the users, which will be shown on the GUI. 
        """

        # Open file explorer 
        file_path = filedialog.askopenfilename()

        # if there is a file path upload it to the GUI and store it
        if file_path:
            self.uploaded_image_path = file_path
            img = Image.open(file_path)
            
            # Resize the image to 640x640
            img = img.resize((640, 640))
            
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def update_image(self, output_image):
        """
        Resuasable Function to save the output image to a predefined path and updates the Tkinter image label.

        Args:
            output_image: the processed image with the custom highlights addeds.  
        
        Output: 
            The output_image, with custom highlights maintaining the attack direction, team roles and processed image. 
        """

        # set the update path for the image
        updated_path = os.path.join(CURRENT_DIR, "output_image.jpg")
        cv2.imwrite(updated_path, output_image)

        # show the updated image on the GUI. 
        updated_img = Image.open(updated_path)

        # Resize the image to 640x640
        updated_img = updated_img.resize((640, 640))
        
        updated_img_tk = ImageTk.PhotoImage(updated_img)
        self.image_label.config(image=updated_img_tk)
        self.image_label.image = updated_img_tk

    def process_and_display_image(self):
        """
        Processes the image using YOLOV8 model and team_assigner to, which is then shown to the user.  

        Output:
            The resulting image, showing bounding boxes, confidence values, classes and masks.  
        """

        # check if the user has uploaded an image. 
        if self.uploaded_image_path:

            # pass image through the image_processer. 
            output_image = processor.process_image(self.uploaded_image_path)
            
            # store the results of the players processed, which is used for custom highlights later. 
            self.processed_players = processor.player_boxes.copy()
            result_path = os.path.join(CURRENT_DIR, "output_image.jpg")
            cv2.imwrite(result_path, output_image)

            # show the resulting image on the GUI. 
            self.result_image_path = result_path
            result_img = Image.open(result_path)

            # Resize the image to 640x640
            result_img = result_img.resize((640, 640))
            result_img_tk = ImageTk.PhotoImage(result_img)
            self.image_label.config(image=result_img_tk)
            self.image_label.image = result_img_tk
            self.assign_roles_frame.pack(pady=10)
        else:
            print("No image uploaded!")
    
    def assign_roles(self):
        """
        Assigns roles of attack and Defence to the teams identifed earlier, alongside the attack direction.  

        Output:
            The processed image with updated labels for the team's role, alongside the attack direction.
        """
        # get team roles attack direction and results. 
        team1_role = self.team1_role_var.get()
        team2_role = self.team2_role_var.get()
        attack_direction = self.attack_direction_var.get()
        results = processor.processed_results
        
        if self.result_image_path:

            # update the visualised image with the corresponding team labels and attack direction. 
            updated_image, _ = visualise_detections(
                cv2.imread(self.result_image_path),
                results,
                processor.model,
                processor.team_assigner,
                PLAYER_CLASS_ID,
                processor.colour_map,
                team1_role,
                team2_role,
                attack_direction,
            )

            self.update_image(updated_image)
        
            # Show the "Offside Detection" button after roles are assigned
            self.offside_detection_button.pack(pady=0)
            
    def pass_data(self): 
        """ 
        Calls the pitch visualiser with processed data and keypoints.

        Output:
            2D Pitch visualisation based on positions of detected classes and keypoints. 
        """
        transformed_footballs = None
        transformed_goalkeepers = None
        transformed_players = None
        transformed_referees = None
        new_football_coordinates = None
        new_goalkeeper_coordinates = None
        new_player_coordinates = None
        new_referee_coordinates = None

        if processor.keypoint_results:
            
            # obtain keypoints and calculate homography matrix. 
            source_pts, valid_indices = position_transformer.normalise_keypoints(processor.keypoint_results)
            H, _ = position_transformer.calculate_homography(source_pts, valid_indices)
            
            # Transform keypoints using the homography matrix
            transformed_points = position_transformer.transform_positions(H, source_pts.reshape(-1, 2))

            print(f"TRANSFORMED KEYPOINTS: {transformed_points}")
        # Transformed Player coordinates     
        if self.processed_players:
            transformed_players = coordinate_transformer.transform_player(self.processed_players)
            
            # Transform positions (apply new transformation to the player coordinates)
            new_player_coordinates = position_transformer.transform_positions(H, transformed_players)
            
            # Append team_colour to each player's transformed coordinates
            new_player_coordinates_with_colour_and_role = coordinate_transformer.assign_roles_and_append_team_colour(new_player_coordinates, self.processed_players, self.team1_role_var.get(), self.team2_role_var.get()) 
            
            print(f"NEW PLAYER COORDINATES WITH COLOUR: {new_player_coordinates_with_colour_and_role}")
        
        # Transformed Referee coordinates     
        if processor.referee_results:
            transformed_referees = coordinate_transformer.transform_referee(processor.referee_results)

            new_referee_coordinates = position_transformer.transform_positions(H, transformed_referees)
            print(f"NEW REFEREE COORDINATES: {new_referee_coordinates}")
        
        # Transformed Goalkeeper coordinates     
        if processor.goalkeeper_results:
            transformed_goalkeepers = coordinate_transformer.transform_goalkeeper(processor.goalkeeper_results)

            new_goalkeeper_coordinates = position_transformer.transform_positions(H, transformed_goalkeepers)
            print(f"NEW GOALKEEPER COORDINATES: {new_goalkeeper_coordinates}")
        
        # Transformed Football coordinates     
        if processor.football_results:
            transformed_footballs = coordinate_transformer.transform_football(processor.football_results)

            new_football_coordinates = position_transformer.transform_positions(H, transformed_footballs)
            print(f"NEW FOOTBALL COORDINATES: {new_football_coordinates}")

        # visualise the pitch
        pitch_display(new_player_coordinates_with_colour_and_role, new_referee_coordinates, new_goalkeeper_coordinates, new_football_coordinates, transformed_points, self.attack_direction_var.get()) 
        
# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
