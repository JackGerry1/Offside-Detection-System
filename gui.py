import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
import os
import cv2
from image_processor import ImageProcessor

# Define paths and color map
CURRENT_DIR = os.getcwd()
MODEL_PATH = f'{CURRENT_DIR}/models/YOLOV8N_SCRATCH_BEST.pt'
COLOUR_MAP = {
    "referee": (0, 0, 0),  # Black
    "football": (0, 165, 255),  # Orange
    "goalkeeper": (255, 105, 180),  # Pink
}

# Initialize ImageProcessor
processor = ImageProcessor(MODEL_PATH, COLOUR_MAP)

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Upload and Processing GUI")
        self.root.geometry("800x950")

        self.uploaded_image_path = None
        self.result_image_path = None

        # Initialize GUI components
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

    def upload_and_display_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.uploaded_image_path = file_path
            img = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def process_and_display_image(self):
        if self.uploaded_image_path:
            output_image = processor.process_image(self.uploaded_image_path)
            result_path = os.path.join(CURRENT_DIR, "output_single_image.jpg")
            cv2.imwrite(result_path, output_image)

            # something here 
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
        attack_direction = self.attack_direction_var.get()  # Get the selected attack direction
        if self.result_image_path:
            updated_image = processor.update_roles(team1_role, team2_role, attack_direction)
            updated_path = os.path.join(CURRENT_DIR, "output_updated_image.jpg")
            cv2.imwrite(updated_path, updated_image)

            updated_img = Image.open(updated_path)
            updated_img_tk = ImageTk.PhotoImage(updated_img)
            self.image_label.config(image=updated_img_tk)
            self.image_label.image = updated_img_tk

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()