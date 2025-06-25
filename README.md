# Offside Detection System Using Computer Vision And Machine Learning
A Football Tkinter GUI, using YOLOV8 for object detection, K-means clustering for player team classification, and tomography for 2D pitch transformation.

## Brief Description & Demo
This project aims to automate offside detection in football using deep learning and image processing techniques. It identifies players, the ball, the referee, and the goalkeeper from individual match screenshots, before classifying them into teams using K-means clustering based on shirt colours. It analyses their positions on a transformed 2D pitch to detect and visualise potential offside, as demonstrated below.
![GUI Screenshot](assets/sjp-xbha-xkz2025-04-0610_19GMT1online-video-cutter.com1-ezgif.com-video-to-gif-converter.gif)

## Installation & Usage

Clone Repo
```bash
git clone https://github.com/yourusername/offside-detection.git
```
Move into Offside-Detection-System Folder
```bash
cd Offside-Detection-System
```
Install the Relevant Libraries

```bash
pip install -r requirements.txt
```
Run Program 
```bash
python main.py
```

## Key Features 
- GUI built with Tkinter for easy interaction
- Upload Chosen Images
- YOLOv8-based object detection players, ball, referee, goalkeeper and keypoints
- K-Means clustering for team classification
- 2D pitch transformation using homography, with MplSoccer Library for Pitch Creation 
- Automated offside line visualisation and decision

## System Architecture 
Below is the architecture diagram representing the system's process, which will be explained in detail in subsequent sections. 
![architecture](assets/architecture.png)

## Benefits

### Detection & Accuracy

- **Players**: 97.73% accuracy  
- **Referees**: 90.19% accuracy  
- **Goalkeepers**: 82.05% accuracy  
- **Team Classification**: 82% accuracy  
- **Keypoint Detection**: 99% precision, 100% recall within the dataset

### Prototype Performance

- **Fast Processing**: ~24.7s per result (faster than traditional VAR systems)
- **Accurate 2D Pitch Mapping**: Reliable when keypoints are well spaced
- **Offside Classification**: Effective in clear, structured scenarios
****

## Limitations

- Football Detection Accuracy ~49%, needs improvement.
- Doesn’t generalise keypoints well to new images or perform accurately when they are close together.
- Struggles with team classification on similar kits and under poor lighting conditions.
- 2D Transformation: Less accurate when keypoints are close together.
- Cannot manually select players if misclassified for offside/onside detection.
- Offside lines are not drawn on the original match image.

---

## Future Work

-  **Upgrade to YOLOv12**
-  Improve goalkeeper/referee distinction (targeted data)
-  Separate model for football detection
-  Enhance team classification with shadow/glare correction
-  Improve offside logic and precision
-  Generalisation & UX enhancements
-  Add user-controlled selection for attackers/defenders
-  Collect broader datasets (new match angles)
-  Use YOLOv12-seg segmentation to detect offside-relevant body parts
-  Redesign the GUI for clarity and ease of use

TODO FINISH THIS: 
# 🚀 **System Breakdown**

This section provides an in-depth explanation of the core components and algorithms used in the Offside Detection System.

---

## 1. Object Detection (YOLO)

### Description:
Object detection identifies various objects on the football pitch such as players, the ball, referees, and goalkeepers using the **YOLO (You Only Look Once)** algorithm. YOLO is a real-time object detection system that provides high accuracy in detecting these key objects.

### How It Works:
- The model is trained on a custom dataset with labeled images of football matches, containing classes such as players, ball, referee, and goalkeeper.
- YOLO uses a single neural network to predict bounding boxes and class probabilities directly from full images in one evaluation.

### Process:
- **Input**: Video frames or images.
- **Output**: Bounding boxes around detected objects (players, ball, etc.).

### Image Example:
![YOLO Example](path_to_image.jpg)

---

## 2. Keypoint Detection

### Description:
Keypoint detection involves identifying and localizing key points (such as knees, elbows, etc.) on players' bodies. This is crucial for determining player positions and calculating offside/onside decisions.

### How It Works:
- Using a **pose estimation model**, we identify keypoints on the player, such as the feet, knees, and torso.
- These keypoints are then used to accurately map the position of the player on the 2D pitch.

### Process:
- **Input**: Video frame or image with a detected player.
- **Output**: Keypoint coordinates (e.g., (x, y) positions for each keypoint).

### Image Example:
![Keypoint Detection](path_to_keypoint_image.jpg)

---

## 3. Team Classification (K-Means Clustering)

### Description:
This algorithm uses **K-Means clustering** to classify players into two teams based on their detected kits, positions, and relative distances from each other.

### How It Works:
- The system first detects the color of kits worn by players.
- Using the **K-Means** algorithm, players are clustered into two groups: one representing the attacking team and the other representing the defending team.

### Process:
- **Input**: Detected players with bounding boxes.
- **Output**: Teams classified into groups (e.g., attacking vs. defending).

### Image Example:
![Team Classification](path_to_team_classification_image.jpg)

---

## 4. 2D Pitch Transformation (Homography)

### Description:
The **2D pitch transformation** is performed using **homography** to create a bird’s-eye view of the football pitch. This helps normalize the field, making it easier to calculate offside positions.

### How It Works:
- Homography is a transformation that maps a 3D perspective to a 2D plane.
- The system identifies key points on the pitch (goal lines, center circle, etc.) and uses these to compute a **homography matrix** to transform the perspective.

### Process:
- **Input**: Image with detected keypoints and football pitch.
- **Output**: Transformed 2D bird’s-eye view of the pitch.

### Image Example:
![Pitch Transformation](path_to_pitch_transformation_image.jpg)

---

## 5. Homography Transformation (Mathematical Explanation)

### Description:
The **homography transformation** allows the system to map the perspective from the camera view to a 2D plane (bird's-eye view), which is essential for accurate offside detection.

### How It Works:
- The transformation is based on a set of corresponding points between the 3D world (football pitch) and the 2D image.
- These points are used to calculate the **homography matrix** that warps the image into a top-down view.

### Mathematical Explanation:
The homography matrix `H` is calculated using the equation:
\[
d = H \cdot p
\]
Where:
- `d` is the 2D coordinates in the transformed plane.
- `p` is the 3D coordinates in the original plane.

### Image Example:
![Homography Transformation](path_to_homography_image.jpg)

---

## 6. Offside/Ondside Detection Algorithm

### Description:
This is the core algorithm that determines whether a player is offside or onside. It uses the 2D transformed pitch and player positions to make the decision.

### How It Works:
- Once the players and keypoints are identified, the system calculates the **last defender line** (the line formed by the last defender in relation to the ball).
- A player is considered **offside** if they are closer to the goal line than the second-to-last defender at the time the ball is played.

### Process:
- **Input**: Transformed 2D pitch, player positions.
- **Output**: Offside or onside decision.

### Image Example:
![Offside Detection](path_to_offside_detection_image.jpg)

---

## 7. Visualization

### Description:
The results of the offside detection algorithm are visualized by overlaying lines on the 2D pitch, indicating the offside line, the positions of players, and whether they are offside or onside.

### How It Works:
- The system uses **Matplotlib** (or similar) to draw:
  - Offside line
  - Player positions
  - Ball location
  - Team classifications

### Process:
- **Input**: 2D transformed pitch, player positions, offside decision.
- **Output**: Image with overlaid visualizations (lines, player markers).

### Image Example:
![Offside Visualization](path_to_visualization_image.jpg)





