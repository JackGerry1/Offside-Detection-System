from roboflow import Roboflow
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

# Use the API key in the Roboflow client to download the training dataset
rf = Roboflow(api_key=roboflow_api_key)
#project = rf.workspace("workspace1-8owu3").project("football-players-yjzti")
##version = project.version(6)
#dataset = version.download("yolov8")

project = rf.workspace("workspace1-8owu3").project("football-field-detection-f07vi-x92hn")
version = project.version(1)
dataset = version.download("yolov8")