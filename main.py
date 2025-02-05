# import neccessary libraries
import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
import os
import cv2
from pitch_visualiser import pitch_display, display_keypoint_data, display_player_data, display_referee_data
from position_transformer import PositionTransformer
import numpy as np
from coordinate_transformer import CoordinateTransformer
# import utilities and other functions
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
        self.root.title("Image Upload and Processing GUI")
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
        self.image_label.bind("<Button-1>", self.on_image_click)
        
        # Role assignment drop down menu for team role and attack direction.
        self.assign_roles_frame = tk.Frame(self.root)
        self.team1_role_var = StringVar(value="Attack")
        self.team2_role_var = StringVar(value="Defense")
        self.attack_direction_var = StringVar(value="Left")  

        # Team role selection
        team1_label = tk.Label(self.assign_roles_frame, text="Team 1 Role:")
        team1_label.grid(row=0, column=0, padx=5, pady=5)
        team1_role_menu = tk.OptionMenu(self.assign_roles_frame, self.team1_role_var, "Attack", "Defense")
        team1_role_menu.grid(row=0, column=1, padx=5, pady=5)

        team2_label = tk.Label(self.assign_roles_frame, text="Team 2 Role:")
        team2_label.grid(row=1, column=0, padx=5, pady=5)
        team2_role_menu = tk.OptionMenu(self.assign_roles_frame, self.team2_role_var, "Attack", "Defense")
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

        # User choice for highlight type
        #self.highlight_choice_var = StringVar(value="Attacker")
        #highlight_label = tk.Label(self.assign_roles_frame, text="Highlight Type:")
        #highlight_label.grid(row=4, column=0, padx=5, pady=5)
        #highlight_menu = tk.OptionMenu(self.assign_roles_frame, self.highlight_choice_var, "Attacker", "Defender")
        #highlight_menu.grid(row=4, column=1, padx=5, pady=5)

        # Create Visualize Pitch button but hide it initially
        self.visualize_pitch_button = tk.Button(self.root, text="Visualize Pitch", command=self.pass_data)
        self.visualize_pitch_button.pack()
        self.visualize_pitch_button.pack_forget() 


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
        updated_path = os.path.join(CURRENT_DIR, "output_updated_image.jpg")
        cv2.imwrite(updated_path, output_image)

        # show the updated image on the GUI. 
        updated_img = Image.open(updated_path)
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
            result_path = os.path.join(CURRENT_DIR, "output_single_image.jpg")
            cv2.imwrite(result_path, output_image)

            # show the resulting image on the GUI. 
            self.result_image_path = result_path
            result_img = Image.open(result_path)
            result_img_tk = ImageTk.PhotoImage(result_img)
            self.image_label.config(image=result_img_tk)
            self.image_label.image = result_img_tk
            self.assign_roles_frame.pack(pady=10)
        else:
            print("No image uploaded!")
    
    def assign_roles(self):
        """
        Assigns roles of attack and defense to the teams identifed earlier, alongside the attack direction.  

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
        
            # **Show the "Visualize Pitch" button after roles are assigned**
            self.visualize_pitch_button.pack(pady=0)
            

    def on_image_click(self, event):
        """
        Detects when a user clicks on the image, verifying if they have clicked on a player.   

        Args:
            event: When the user presses Mouse1 
        
        References: 
            Leekha, G. (2023). How to find tags near to mouse click in Tkinter Python GUI? [online] Tutorialspoint.com. 
            Available at: https://www.tutorialspoint.com/how-to-find-tags-near-to-mouse-click-in-tkinter-python-gui [Accessed 16 Jan. 2025].
        
        Output: Highlighted Box around the player that the user clicked, using styling from the highlight_player function. 
        """
        
        # check if there is an image or players detected. 
        if not self.result_image_path or not self.processed_players:
            print("No players to select.")
            return

        # obtain image and label, width and height. 
        # img width and height is 640
        # label width and height is 644, so just a bit of padding around the image.  
        img_width, img_height = Image.open(self.result_image_path).size
        label_width, label_height = self.image_label.winfo_width(), self.image_label.winfo_height()

        # get scale value for x and y
        scale_x = img_width / label_width
        scale_y = img_height / label_height

        # detect coordinate of user's click. 
        x_click = int(event.x * scale_x)
        y_click = int(event.y * scale_y)
        print(f"YOU CLICKED AT: {x_click}, {y_click}")
        # loop through all processed players and compare the user's click to their bounding boxes, 
        # If they click on a player it will highlight them. 
        for player in self.processed_players:  
            x_min, y_min, x_max, y_max = map(int, player["coords"])
            if x_min <= x_click <= x_max and y_min <= y_click <= y_max:
                self.highlight_player(player)
                break
        else:
            print("No player clicked.")


    def highlight_player(self, player):
        """
        Highlights the player, which the user clicked and draws a bounding box around them.  

        Args:
            player: the player's information, such as coordinate and team, which the user has clicked on. 
        
        Output: Highlighted Box around the player that the user clicked, alongside a corresponding label. 
        """ 

        # obtoain highlight choices, assign highlight_key and get results about the image. 
        highlight_choice = self.highlight_choice_var.get()
        highlight_key = "FA" if highlight_choice == "Attacker" else "FBD"
        results = processor.processed_results

        # apply the custom highlight to the obtained player.  
        self.custom_highlights[highlight_key] = player
        
        print(f"Custom {highlight_key} updated: {player}")  

        # update the visualisation with the newly highlighted player, while maintaing all information from previous visualisations.  
        if self.result_image_path:
            output_image, _ = visualise_detections(
                cv2.imread(self.result_image_path),
                results,
                processor.model,
                processor.team_assigner,
                PLAYER_CLASS_ID,
                processor.colour_map,
                self.team1_role_var.get(),
                self.team2_role_var.get(),
                self.attack_direction_var.get(),
                custom_highlights=self.custom_highlights
            )

            self.update_image(output_image)
    
    # pass data about keypoints and detected, refs, players, goalkeepers and footballs. 
    def pass_data(self): 
        """ Calls the pitch visualiser with processed player data and keypoints, only if data is available. """
        transformed_footballs = None
        transformed_goalkeepers = None
        transformed_players = None
        transformed_referees = None
        new_football_coordinates = None
        new_goalkeeper_coordinates = None
        new_player_coordinates = None
        new_referee_coordinates = None
        
        if processor.keypoint_results:
            
            display_keypoint_data(processor.keypoint_results)

            source_pts, valid_indices = position_transformer.normalise_keypoints(processor.keypoint_results)
            H, _ = position_transformer.calculate_homography(source_pts, valid_indices)
            
            #print(f"HOMOGRAPHY: {H}")

            # Transform keypoints using the homography matrix
            transformed_points = position_transformer.transform_positions(H, source_pts.reshape(-1, 2))

            print(f"TRANSFORMED KEYPOINTS: {transformed_points}")
            
        if self.processed_players:
            display_player_data(self.processed_players, self.team1_role_var.get(), self.team2_role_var.get())
            
            # Transform player coordinates
            transformed_players = coordinate_transformer.transform_player(self.processed_players)
            
            # Transform positions (apply new transformation to the player coordinates)
            new_player_coordinates = position_transformer.transform_positions(H, transformed_players)
            
            # Append team_colour to each player's transformed coordinates
            new_player_coordinates_with_colour = [
                np.append(new_coordinates, player['team_colour'])
                for new_coordinates, player in zip(new_player_coordinates, self.processed_players)
            ]
            
            # Convert the list to a NumPy array
            new_player_coordinates_with_colour = np.array(new_player_coordinates_with_colour)
            
            print(f"NEW PLAYER COORDINATES WITH COLOUR: {new_player_coordinates_with_colour}")

        if processor.referee_results:
            transformed_referees = coordinate_transformer.transform_referee(processor.referee_results)
            #print("REFEREE RESULTS: " + str(transformed_referees))

            new_referee_coordinates = position_transformer.transform_positions(H, transformed_referees)
            print(f"NEW REFEREE COORDINATES: {new_referee_coordinates}")

        if processor.goalkeeper_results:
            transformed_goalkeepers = coordinate_transformer.transform_goalkeeper(processor.goalkeeper_results)
            #print("GOALKEEPER RESULTS: " + str(transformed_goalkeepers))

            new_goalkeeper_coordinates = position_transformer.transform_positions(H, transformed_goalkeepers)
            print(f"NEW GOALKEEPER COORDINATES: {new_goalkeeper_coordinates}")

        if processor.football_results:
            transformed_footballs = coordinate_transformer.transform_football(processor.football_results)
            #print("FOOTBALL RESULTS: " + str(transformed_footballs)) 

            new_football_coordinates = position_transformer.transform_positions(H, transformed_footballs)
            print(f"NEW FOOTBALL COORDINATES: {new_football_coordinates}")

        pitch_display(new_player_coordinates_with_colour, new_referee_coordinates, new_goalkeeper_coordinates, new_football_coordinates, transformed_points) 


        
        
# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
