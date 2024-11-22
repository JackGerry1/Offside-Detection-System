from roboflow import Roboflow
rf = Roboflow(api_key="oxh0PmyBG9tLQTZYozrm")
project = rf.workspace("workspace1-8owu3").project("football-players-yjzti")
version = project.version(6)
dataset = version.download("yolov8")
                