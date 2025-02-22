import numpy as np
from utils.utils import PITCH_LENGTH, PITCH_WIDTH

class CoordinateTransformer:
    def __init__(self):
        self.transformed_coordinates = []
        self.image_size = 640

    def transform_coordinates(self, x_min, _, x_max, y_max):
        """
        Transforms bounding box coordinates to bottom center coordinates

        Parameters:
        - x_min: top left corner of bounding box
        - _: y_min top of bounding box
        - x_max: top right corner of bounding box
        - y_max: bottom of bounding box

        Returns:
        - X and Y normalised coordinates. 
        """
        # get bottom center coordinates 
        x_center = (x_min + x_max) / 2
        y_center = y_max 

        # normalises to size of 2D pitch. 
        x_normalised = x_center * (PITCH_LENGTH / self.image_size)  
        y_normalised = y_center * (PITCH_WIDTH / self.image_size)  
        return np.array([x_normalised, y_normalised]) 
    
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
        """
        Transforms the players coordinates into normalised x and y values. 

        Parameters:
        - players: player coordinates

        Returns:
        - Transformed player coordinates in np.array
        """
        transformed_players = [self.transform_coordinates(*player['coords']) for player in players]
        return np.array(transformed_players)  # Convert list of arrays to NumPy array (N, 2)

    def transform_referee(self, referees):
        """
        Transforms the referee coordinates into normalised x and y values. 

        Parameters:
        - referee: referee coordinates

        Returns:
        - Transformed referee coordinates in np.array
        """
        transformed_referees = [self.transform_coordinates(*referee) for referee in referees]
        return np.array(transformed_referees)

    def transform_football(self, footballs):
        """
        Transforms the football coordinates into normalised x and y values. 

        Parameters:
        - footballs: football coordinates

        Returns:
        - Transformed football coordinates in np.array
        """
        transformed_footballs = [self.transform_coordinates(*football) for football in footballs]
        return np.array(transformed_footballs)

    def transform_goalkeeper(self, goalkeepers):
        """
        Transforms the goalkeepers coordinates into normalised x and y values. 

        Parameters:
        - goalkeepers: goalkeeper coordinates

        Returns:
        - Transformed goalkeeper coordinates in np.array
        """
        transformed_goalkeepers = [self.transform_coordinates(*goalkeeper) for goalkeeper in goalkeepers]
        return np.array(transformed_goalkeepers)
