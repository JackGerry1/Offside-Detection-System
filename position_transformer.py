import numpy as np
import cv2
from utils.utils import CONFIG_VERTICES, PITCH_WIDTH, PITCH_LENGTH

class PositionTransformer:
    def __init__(self):
        self.image_size = 640
    def normalise_keypoints(self, keypoints):
        """
        Normalises keypoints from the source image (640x640) 
        to match the dimensions of the 2D pitch (110m x 69m).
        
        Filters out invalid keypoints (e.g., confidence < 0.5).
        Returns both the normalised keypoints and their valid indices.
        """
        source_pts = []
        valid_indices = []
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.5:  # Keep only high-confidence keypoints
                x_normalised = x * (PITCH_LENGTH / self.image_size)  # Scale x-coordinate
                y_normalised = y * (PITCH_WIDTH / self.image_size)  # Flip Y and scale to match pitch
                source_pts.append([x_normalised, y_normalised])
                valid_indices.append(i)  # Store the index of the valid keypoint

        return np.array(source_pts, dtype=np.float32).reshape(-1, 1, 2), valid_indices
    
    def calculate_homography(self, source_pts, valid_indices):
        """
        Computes homography matrix using filtered keypoints.
        Only keeps target keypoints corresponding to valid source keypoints.
        """
        target_pts = np.array(CONFIG_VERTICES, dtype=np.float32)  # Load target keypoints
        
        if len(valid_indices) == 0:
            raise ValueError("No valid keypoints to compute homography.")

        # Filter target keypoints to match the valid source keypoints
        filtered_target_pts = target_pts[valid_indices]
        filtered_target_pts = filtered_target_pts.reshape(-1, 1, 2)

        # Ensure source and target keypoints have the same shape
        if source_pts.shape != filtered_target_pts.shape:
            raise ValueError("Source and target keypoints must have the same shape.")

        # Compute Homography Matrix
        H, status = cv2.findHomography(source_pts, filtered_target_pts, cv2.RANSAC, 5.0)
        return H, status

    def transform_positions(self, H, positions):
        """
        Applies a perspective transformation to a set of positions using a homography matrix.
        """
        positions_reshaped = positions.reshape(-1, 1, 2).astype(np.float32)

        transformed_positions = cv2.perspectiveTransform(positions_reshaped, H)

        return transformed_positions.reshape(-1, 2).astype(np.float32)