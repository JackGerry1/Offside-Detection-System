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

def pitch_display(players=None, referees=None, goalkeepers=None, footballs=None, transformed_points=None): 
    # Initialise pitch with updated dimensions
    pitch = Pitch(pitch_type='custom', pitch_width=PITCH_WIDTH, pitch_length=PITCH_LENGTH, 
                  goal_type='box', linewidth=2, line_color='white', pitch_color='green')

    # Create figure
    _, ax = pitch.draw(figsize=(10, 6))


    print(COLOUR_MAP["referee"])
    print(COLOUR_MAP["football"])
    print(COLOUR_MAP["goalkeeper"])
    # Reverse the y-axis so that (0,0) is at the top-left
    ax.invert_yaxis()
    
    # Plot pitch markers
    #ax.scatter(CONFIG_VERTICES[:, 0], CONFIG_VERTICES[:, 1], color='yellow', s=150, edgecolors='black', zorder=3, label='Pitch Markers')

    # Plot transformed points if available
    if transformed_points is not None and transformed_points.size > 0.1:
        ax.scatter(transformed_points[:, 0], transformed_points[:, 1], color='yellow', s=100, edgecolors='black', zorder=4, label='Transformed Points')

    # Plot transformed players with dynamic colors
    if players is not None:
        player_positions = players[:, :2]  # Extract x and y positions
        player_colours = players[:, 2:5] / 255
        
        # Scatter plot with dynamic colors
        ax.scatter(player_positions[:, 0], player_positions[:, 1], color=player_colours, s=100, edgecolors='black', label='Players') 

    # Plot transformed referees if available
    if referees is not None:
        ax.scatter(referees[:, 0], referees[:, 1], color='black', s=100, edgecolors='black', label='Referees')

    # Plot transformed goalkeepers if available
    if goalkeepers is not None:
        ax.scatter(goalkeepers[:, 0], goalkeepers[:, 1], color='pink', s=100, edgecolors='black', label='Goalkeepers')

    # Plot transformed footballs if available
    if footballs is not None:
        ax.scatter(footballs[:, 0], footballs[:, 1], color='white', s=100, edgecolors='black', label='Football')

    plt.show()

if __name__ == "__main__":
    pitch_display(players=None, referees=None, goalkeepers=None, footballs=None, transformed_points=None)