from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np

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

# TODO: apply filter to get the matching config vertices and keypoints.  

def main(): 
    # Initialise pitch with updated dimensions
    pitch = Pitch(pitch_type='custom', pitch_width=69, pitch_length=110, 
                  goal_type='box', linewidth=2, line_color='white', pitch_color='green')

    # Create figure
    _, ax = pitch.draw(figsize=(10, 6))

    # Define points corresponding to the new configuration in meters
    CONFIG_VERTICES = np.array([
        (0, 0),         # Bottom-left corner of the pitch 
        (0, 14.5),      # Bottom Left Of Box 
        (0, 25.25),     # Bottom Left Of Six Yard Box
        (0, 43.75),     # Top Left Of Six Yard Box
        (0, 54.75),     # Top Left Of Box
        (0, 69),        # Top-left corner of the pitch
        (5.5, 25.25),   # Left Bottom six-yard box outside edge 
        (5.5, 43.75),   # Left Top six-yard box outside edge 
        (11, 34.5),     # Left penalty spot
        (16.5, 14.5),   # Outside Bottom Left box
        (16.5, 27.5),   # Left Penaltiy Arc Bottom
        (16.5, 41),     # Left Penaltiy Arc Top
        (16.5, 54.75),  # Outside Top Left box
        (55, 0),        # In line with bottom of centre circle but at the bottom of the pitch
        (55, 25.25),    # Centre circle bottom edge
        (55, 43.75),    # Centre circle top edge
        (45.75, 34.5),  # Centre circle left side
        (64, 34.5),     # Centre circle right side
        (55, 69),       # In line with top of centre circle but at the top of the pitch
        (93.5, 14.5),   # Outside Bottom Right box
        (93.5, 28),     # Right Penaltiy Arc Bottom
        (93.5, 41),     # Right Penaltiy Arc Top
        (93.5, 54.75),  # Outside Top Right box
        (99, 34.5),     # Right penalty spot
        (110, 0),       # Bottom-right corner of the pitch
        (110, 14.5),    # Bottom Right box
        (110, 25.25),   # Bottom Right Of Six Yard Box
        (110, 43.75),   # Top Right Of Six Yard Box
        (104.5, 25.25), # Right Bottom six-yard box outside edge
        (104.5, 43.75), # Right Top six-yard box outside edge
        (110, 54.75),   # Top Right Of Box
        (110, 69),      # Top-right corner of the pitch
    ])

    # Randomly generate positions for players, goalkeepers, referee, and football
    num_players_per_team = 5
    team1_positions = np.random.rand(num_players_per_team, 2) * [110, 69]
    team2_positions = np.random.rand(num_players_per_team, 2) * [110, 69]
    goalkeepers = np.array([[5.5, 34.5], [104.5, 34.5]])  # Approximate goalkeeper positions
    referee = np.array([[55, 34.5]])  # Referee positioned at centre circle
    football = np.array([[55, 25]])  # Football placed near the centre

    # Plot points
    ax.scatter(CONFIG_VERTICES[:, 0], CONFIG_VERTICES[:, 1], color='yellow', s=150, edgecolors='black', zorder=3, label='Pitch Markers')

    # Display plot
    plt.show()

if __name__ == "__main__":
    main()