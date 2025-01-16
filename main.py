import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
import os
import cv2
from image_processing.image_processor import ImageProcessor
from visualisation.visualise import visualise_detections
from utils.utils import draw_bounding_box_with_label, CURRENT_DIR, MODEL_PATH, COLOUR_MAP

# Initialize ImageProcessor
processor = ImageProcessor(MODEL_PATH, COLOUR_MAP)

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Upload and Processing GUI")
        self.root.geometry("800x950")

        self.uploaded_image_path = None
        self.result_image_path = None
        self.custom_highlights = {"FA": None, "FBD": None}
        self.processed_players = []

        # Initialise GUI components
        self.init_components()

    def init_components(self):
        # Upload button
        upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_and_display_image)
        upload_button.pack(pady=10)

        # Process button
        process_button = tk.Button(self.root, text="Process Image", command=self.process_and_display_image)
        process_button.pack(pady=10)

        # Image label
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.on_image_click)
        # Role assignment frame
        self.assign_roles_frame = tk.Frame(self.root)
        self.team1_role_var = StringVar(value="Attack")
        self.team2_role_var = StringVar(value="Defense")
        self.attack_direction_var = StringVar(value="Left")  # Default attack direction

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
        file_path = filedialog.askopenfilename()
        if file_path:
            self.uploaded_image_path = file_path
            img = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def update_image(self, output_image):
        """
        Saves the output image to a predefined path and updates the Tkinter image label.

        Args:
            output_image: The image to be saved and displayed.
        """
        updated_path = os.path.join(CURRENT_DIR, "output_updated_image.jpg")
        cv2.imwrite(updated_path, output_image)

        updated_img = Image.open(updated_path)
        updated_img_tk = ImageTk.PhotoImage(updated_img)
        self.image_label.config(image=updated_img_tk)
        self.image_label.image = updated_img_tk

    def process_and_display_image(self):
        if self.uploaded_image_path:
            output_image = processor.process_image(self.uploaded_image_path)
            self.processed_players = processor.player_boxes.copy()  # Store the initial player data
            result_path = os.path.join(CURRENT_DIR, "output_single_image.jpg")
            cv2.imwrite(result_path, output_image)

            self.result_image_path = result_path
            result_img = Image.open(result_path)
            result_img_tk = ImageTk.PhotoImage(result_img)
            self.image_label.config(image=result_img_tk)
            self.image_label.image = result_img_tk
            self.assign_roles_frame.pack(pady=10)
        else:
            print("No image uploaded!")

    def assign_roles(self):
        team1_role = self.team1_role_var.get()
        team2_role = self.team2_role_var.get()
        attack_direction = self.attack_direction_var.get()
        results = processor.processed_results

        if self.result_image_path:
            # Update the roles in processed_players
            for player in self.processed_players:
                player['role'] = team1_role if player['team'] == 1 else team2_role

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
        if not self.result_image_path or not self.processed_players:
            print("No players to select.")
            return

        img_width, img_height = Image.open(self.result_image_path).size
        label_width, label_height = self.image_label.winfo_width(), self.image_label.winfo_height()

        scale_x = img_width / label_width
        scale_y = img_height / label_height

        x_click = int(event.x * scale_x)
        y_click = int(event.y * scale_y)

        for player in self.processed_players:  # Use processed_players instead of processor.player_boxes
            x_min, y_min, x_max, y_max = map(int, player["coords"])
            if x_min <= x_click <= x_max and y_min <= y_click <= y_max:
                self.highlight_player(player)
                break
        else:
            print("No player clicked.")


    def highlight_player(self, player):
        highlight_choice = self.highlight_choice_var.get()
        highlight_key = "FA" if highlight_choice == "Attacker" else "FBD"
        results = processor.processed_results
        
        self.custom_highlights[highlight_key] = player
        
        print(f"Custom {highlight_key} updated: {player}")  # This will show the role
        
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


    def _remove_highlight(self, _, highlight_type):
        """
        Removes the highlight for the given player by resetting the base image and redrawing other highlights.
        """
        # Reload the base image
        self.highlighted_image = cv2.imread(self.result_image_path)

        # Redraw the other highlight if it exists
        other_key = "FBD" if highlight_type == "FA" else "FA"
        if self.current_highlights[other_key]:
            self._draw_highlight(self.current_highlights[other_key], other_key)

    def _draw_highlight(self, player, highlight_type):
        """
        Draws the bounding box and label for a player on the image.
        """
        colour = (0, 255, 0) if highlight_type == "FA" else (255, 0, 0)
        label = "FA" if highlight_type == "FA" else "FBD"
        draw_bounding_box_with_label(self.highlighted_image, player["coords"], colour, label)


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
