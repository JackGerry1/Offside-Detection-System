from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from utils.utils import CONFIG_VERTICES, PITCH_WIDTH, PITCH_LENGTH, COLOUR_MAP

def display_player_data(processed_players, team1_role, team2_role):
    for player in processed_players:
                player['role'] = team1_role if player['team'] == 1 else team2_role
                print(f"PLAYER: {player}")

def display_referee_data(referees):
    for referee in referees:
        print(f"REFEREE: {referee}") 

def display_goalkeeper_data(goalkeepers):
    for goalkeeper in goalkeepers:
        print(f"GOALKEEPER: {goalkeeper}")  

def display_football_data(footballs):
    for football in footballs:
        print(f"FOOTBALL: {football}")   
    
def display_keypoint_data(keypoints): 
    # Extract keypoints and print them
    for idx, keypoint in enumerate(keypoints):
        x, y, confidence = keypoint[0], keypoint[1], keypoint[2]
        print(f"Keypoint {idx + 1}: X: {x:.2f}, Y: {y:.2f}, Confidence: {confidence:.4f}")

def find_relevant_players(players, attack_direction):
    """
    Find furthest forward attacker and back defender from player coordinates.
    """
    direction_multiplier = 1 if attack_direction.lower() == "right" else -1
    
    # Initialise extremes
    max_attack_x = -float('inf')
    max_defend_x = -float('inf')
    forward_player = None
    back_player = None
    
    for idx, player in enumerate(players):
        x = float(player[0])
        y = float(player[1])
        role = player[5]
        
        # Compare x-positions based on direction
        pos_value = direction_multiplier * x
        
        if role == "Attack" and pos_value > max_attack_x:
            max_attack_x = pos_value
            forward_player = (x, y)
            
        if role == "Defense" and pos_value > max_defend_x:
            max_defend_x = pos_value
            back_player = (x, y)
            
    return forward_player, back_player     

def pitch_display(players=None, referees=None, goalkeepers=None, footballs=None, transformed_points=None, attack_direction=None): 
    # Initialise pitch with updated dimensions
    pitch = Pitch(pitch_type='custom', pitch_width=PITCH_WIDTH, pitch_length=PITCH_LENGTH, 
                  goal_type='box', linewidth=2, line_color='white', pitch_color='green')
    print(f"pitch visualiser attack direction: {attack_direction}")
    print(f"pitch visualiser player: {players}")
    # Create figure
    _, ax = pitch.draw(figsize=(10, 6))

    ref_colour = np.flip(np.array(COLOUR_MAP["referee"]) / 255)
    football_colour = np.flip(np.array(COLOUR_MAP["football"]) / 255)
    goalkeeper_colour = np.flip(np.array(COLOUR_MAP["goalkeeper"]) / 255)

    # Reverse the y-axis so that (0,0) is at the top-left
    ax.invert_yaxis()
    
    # Plot pitch markers
    #ax.scatter(CONFIG_VERTICES[:, 0], CONFIG_VERTICES[:, 1], color='yellow', s=150, edgecolors='black', zorder=3, label='Pitch Markers')

    # Plot transformed points if available
    #if transformed_points is not None and transformed_points.size > 0.1:
    #    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], color='yellow', s=100, edgecolors='black', zorder=4, label='Transformed Points')
    
    # Plot players
    if players is not None:
        player_positions = np.float32(players[:, :2])
        player_colours = np.float32(players[:, 2:5]) / 255
        
        # Find extreme players
        forward_player, back_player = find_relevant_players(players, attack_direction)

        forward_player = np.float32(forward_player)
        back_player = np.float32(back_player)

        # Plot base player points
        ax.scatter(player_positions[:, 0], player_positions[:, 1], 
                  color=player_colours, s=100, edgecolors='black', label='Players')
        
        # Add highlight circles
        if forward_player is not None:
            ax.scatter(forward_player[0], forward_player[1], color='#FFFF00', s=100, edgecolors='white')
            
        if back_player is not None:
            ax.scatter(back_player[0], back_player[1], color='#FF0000', s=100, edgecolors='white')

    # Plot transformed referees if available
    if referees is not None:
        ax.scatter(referees[:, 0], referees[:, 1], color=ref_colour, s=100, edgecolors='black', label='Referees')

    # Plot transformed goalkeepers if available
    if goalkeepers is not None:
        ax.scatter(goalkeepers[:, 0], goalkeepers[:, 1], color=goalkeeper_colour, s=100, edgecolors='black', label='Goalkeepers')

    # Plot transformed footballs if available
    if footballs is not None:
        ax.scatter(footballs[:, 0], footballs[:, 1], color=football_colour, s=100, edgecolors='black', label='Football')

    plt.show()

if __name__ == "__main__":
    pitch_display(players=None, referees=None, goalkeepers=None, footballs=None, transformed_points=None)