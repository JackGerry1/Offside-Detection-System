import numpy as np
from utils.utils import PITCH_LENGTH, PITCH_WIDTH

class CoordinateTransformer:
    def __init__(self):
        self.transformed_coordinates = []
        self.image_size = 640

    def transform_coordinates(self, x_min, _, x_max, y_max):
        x_center = (x_min + x_max) / 2
        y_center = y_max 

        x_normalised = x_center * (PITCH_LENGTH / self.image_size)  
        y_normalised = y_center * (PITCH_WIDTH / self.image_size)  
        return np.array([x_normalised, y_normalised])  # Ensure a NumPy array output
    
    def assign_roles_and_append_team_colour(self, new_player_coordinates, processed_players, team1_role, team2_role):
        """
        Assigns roles to players based on their team and appends the flipped team color and role
        to each player's new coordinates.

        Parameters:
        - new_player_coordinates (array-like): A list or array of new player coordinates.
        - processed_players (list of dicts): A list of players, where each player has a 'team' and 'team_colour'.
        - team1_role (str): Role for team 1 (e.g., "Attack").
        - team2_role (str): Role for team 2 (e.g., "Defense").

        Returns:
        - numpy.ndarray: A NumPy array with coordinates, flipped team color, and assigned role.
        """
        new_player_coordinates_with_colour_and_role = []

        for new_coordinates, player in zip(new_player_coordinates, processed_players):
            # Assign the role based on the team
            player['role'] = team1_role if player['team'] == 1 else team2_role

            # Append flipped team color and role
            combined_data = np.append(np.append(new_coordinates, np.flip(player['team_colour'])), player['role'])
            new_player_coordinates_with_colour_and_role.append(combined_data)

        return np.array(new_player_coordinates_with_colour_and_role)

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
