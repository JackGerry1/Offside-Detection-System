'''
References: 
Tarek, A. (2024). Football_Analysis Team Analysis Code. [online] GitHub. 
Available at: https://github.com/abdullahtarek/football_analysis/blob/main/team_assigner/team_assigner.py [Accessed 11 Dec. 2024].

'''

import numpy as np
from sklearn.cluster import KMeans
import cv2 

class TeamAssigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """
        Get Team Shirt Colour For Each Player

        Args:
            image: Bounding box of players detected. 
        Outputs:
            Cluster of team colour values, separated from the background
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def enhance_colour(self, colour):
        """
        Enhance the detected player shirt colour to make classification into two teams. 

        Args:
            colour: player recongised team colour
        Outputs:
            enhanced colour output for the classified team colours. 
        """
        # Strengthen the highest value and weaken smaller ones
        max_val = np.max(colour)
        modified_colour = np.array([
            val * 1.2 if val == max_val else val * 0.8
            for val in colour
        ])
        # Clip to valid RGB range
        return np.clip(modified_colour, 0, 255)

    def get_player_colour(self, image, bbox):
        """
        Obtain Player Shirt Colour

        Args:
            image: player image
            bbox: bounding box of player.  
        Outputs:
            Enhanced Colour for each of the players. 
        """
        # Extract the bounding box region from the image
        x_min, y_min, x_max, y_max = map(int, bbox)
        image = image[y_min:y_max, x_min:x_max]
        
        # resize the image for consistency
        image_resized = cv2.resize(image, (64, 64))
        
        # Focus on the top half of the resized bounding box for shirt colour
        top_half_image = image_resized[:image_resized.shape[0] // 2, :]

        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine the player's dominant cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get the dominant colour for the player's cluster
        player_colour = kmeans.cluster_centers_[player_cluster]

        # Enhance the colour
        enhanced_colour = self.enhance_colour(player_colour)

        return enhanced_colour

    def assign_team_colour(self, image, player_detections):
        """
        Assign two team colours based on shirt colour

        Args:
            image: Uploded Image of Football Match
            player_detections: array of detected players
        Outputs:
            Two teams classifed using k-means clustering. 
        """
        # Extract player colours from all detections
        player_colours = []
        for bbox in player_detections:
            player_colour = self.get_player_colour(image, bbox)
            player_colours.append(player_colour)
        
        # Perform clustering on player colours to determine team colours
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto")
        kmeans.fit(player_colours)

        self.kmeans = kmeans
        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]
        print(self.team_colours[1])
        print(self.team_colours[2])
    def get_player_team(self, image, bbox, player_id):
        """
        Assign players into team 1 and team 2. 

        Args:
            image: uploaded image of a football match 
            bbox: Player detection bounding box. 
            player_id: id of that players corresponding bounding box
        Outputs:
            Team_id for each player. 
        """
        # If already assigned, return the team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extract the player's dominant colour
        player_colour = self.get_player_colour(image, bbox)

        # Predict the team using the KMeans model
        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0]
        team_id += 1  # Convert to 1-based index (1 or 2)

        # Store and return the assigned team
        self.player_team_dict[player_id] = team_id
        return team_id
