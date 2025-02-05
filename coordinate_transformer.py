import numpy as np
from utils.utils import PITCH_LENGTH, PITCH_WIDTH

class CoordinateTransformer:
    def __init__(self):
        self.transformed_coordinates = []
        self.image_size = 640

    def transform_coordinates(self, x_min, y_min, x_max, y_max):
        x_center = (x_min + x_max) / 2
       # y_center = (y_min + y_max) / 2
        y_center = y_max 

        x_normalised = x_center * (PITCH_LENGTH / self.image_size)  
        y_normalised = y_center * (PITCH_WIDTH / self.image_size)  
        return np.array([x_normalised, y_normalised])  # Ensure a NumPy array output

    def transform_player(self, players):
        transformed_players = [self.transform_coordinates(*player['coords']) for player in players]
        return np.array(transformed_players)  # Convert list of arrays to NumPy array (N, 2)

    def transform_referee(self, referees):
        transformed_referees = [self.transform_coordinates(*referee) for referee in referees]
        return np.array(transformed_referees)

    def transform_football(self, footballs):
        transformed_footballs = [self.transform_coordinates(*football) for football in footballs]
        return np.array(transformed_footballs)

    def transform_goalkeeper(self, goalkeepers):
        transformed_goalkeepers = [self.transform_coordinates(*goalkeeper) for goalkeeper in goalkeepers]
        return np.array(transformed_goalkeepers)
