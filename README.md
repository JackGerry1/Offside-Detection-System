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




