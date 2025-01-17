# import neccessary libraries
import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
import os
import cv2

# import utilities and other functions
from image_processing.image_processor import ImageProcessor
from visualisation.visualise import visualise_detections
from utils.utils import CURRENT_DIR, MODEL_PATH, COLOUR_MAP

# Initialise ImageProcessor
processor = ImageProcessor(MODEL_PATH, COLOUR_MAP)

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
        self.highlight_choice_var = StringVar(value="Attacker")
        highlight_label = tk.Label(self.assign_roles_frame, text="Highlight Type:")
        highlight_label.grid(row=4, column=0, padx=5, pady=5)
        highlight_menu = tk.OptionMenu(self.assign_roles_frame, self.highlight_choice_var, "Attacker", "Defender")
        highlight_menu.grid(row=4, column=1, padx=5, pady=5)

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
            # Update the roles in processed_players
            for player in self.processed_players:
                player['role'] = team1_role if player['team'] == 1 else team2_role

            # update the visualised image with the corresponding team labels and attack direction. 
            updated_image = visualise_detections(
                cv2.imread(self.result_image_path),
                results,
                processor.model,
                processor.team_assigner,
                processor.player_class_id,
                processor.colour_map,
                team1_role,
                team2_role,
                attack_direction,
            )

            self.update_image(updated_image)

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
            output_image = visualise_detections(
                cv2.imread(self.result_image_path),
                results,
                processor.model,
                processor.team_assigner,
                processor.player_class_id,
                processor.colour_map,
                self.team1_role_var.get(),
                self.team2_role_var.get(),
                self.attack_direction_var.get(),
                custom_highlights=self.custom_highlights
            )

            self.update_image(output_image)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
