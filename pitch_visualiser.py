from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import CONFIG_VERTICES, PITCH_WIDTH, PITCH_LENGTH, COLOUR_MAP

def find_relevant_players(players, attack_direction):
    """
    Find furthest forward attacker and back defender from player coordinates.

    Output: 
        Coordinates of furthest forward attacker and furthest back defender. 
    """
    direction_multiplier = 1 if attack_direction.lower() == "right" else -1
    
    # Initialise extremes
    max_attack_x = -float('inf')
    max_defend_x = -float('inf')
    forward_player = None
    back_player = None
    
    for player in players:
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
            
    return np.float32(forward_player), np.float32(back_player)     


def check_offside(forward_player, back_player, attack_direction):
    """
    Determines if the attacker is offside.

    Args: 
        forward_player: x y coordinates of attacker 
        back_player: x y coordinates of defender
        attack_direction: left or right specified by the user

    Output:
        Returns the color (red for offside, green for onside).
    """
    
    direction_multiplier = 1 if attack_direction.lower() == "right" else -1
    is_offside = (direction_multiplier * forward_player[0]) > (direction_multiplier * back_player[0])
    
    # red or light green
    return '#FF0000' if is_offside else '#00FF00'

def pitch_display(players=None, referees=None, goalkeepers=None, footballs=None, transformed_points=None, attack_direction=None):
    """
    Displays pitch based on data paramters from the source image. 

    Args: 
        players: transformed coordinates of players 
        referees: transformed coordinates of referees 
        goalkeepers: transformed coordinates of goalkeepers 
        footballs: transformed coordinates of footballs 
        transformed_points: keypoints transformed positions 
        attack_direction: left or right. 

    References: 
    Durgapal, A. and Rowlinson, A. (2021a). Pitch Basics — Mplsoccer 1.4.0 Documentation. [online] Readthedocs.io. 
    Available at: https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html#sphx-glr-gallery-pitch-setup-plot-pitches-py [Accessed 21 Feb. 2025].

    Durgapal, A. and Rowlinson, A. (2021b). Scatter — Mplsoccer 1.4.0 Documentation. [online] Readthedocs.io. 
    Available at: https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_scatter.html [Accessed 21 Feb. 2025].

    Output: 
        2D transformed pitch with relevant objects visualised accurately. 
    """
    # Initialise pitch with updated dimensions
    pitch = Pitch(pitch_type='custom', pitch_width=PITCH_WIDTH, pitch_length=PITCH_LENGTH, 
                  goal_type='box', linewidth=2, line_color='#E0E0E0', pitch_color='#1A1A1D')
    
    # Create figure
    fig, ax = pitch.draw(figsize=(10, 6))

    ref_colour = np.flip(np.array(COLOUR_MAP["referee"]) / 255)
    football_colour = np.flip(np.array(COLOUR_MAP["football"]) / 255)
    goalkeeper_colour = np.flip(np.array(COLOUR_MAP["goalkeeper"]) / 255)
    marker_size = 100
    
    # Reverse the y-axis so that (0,0) is at the top-left
    ax.invert_yaxis()
    
    # Plot pitch markers
    #ax.scatter(CONFIG_VERTICES[:, 0], CONFIG_VERTICES[:, 1], color='yellow', s=marker_size, edgecolors='white', zorder=3, label='Pitch Markers')

    # Plot transformed points if available
    #if transformed_points is not None and transformed_points.size > 0.1:
    #    ax.scatter(transformed_points[:, 0], transformed_points[:, 1], color='yellow', s=mark_size, edgecolors='white', zorder=4, label='Transformed Points')
    
    # Plot players
    if players is not None:
        player_positions = np.float32(players[:, :2])
        player_colours = np.float32(players[:, 2:5]) / 255
        
        forward_player, back_player = find_relevant_players(players, attack_direction)
        print(f"Forward Player: {forward_player}, Back Player: {back_player}")
        
        attacker_colour = check_offside(forward_player, back_player, attack_direction)
        
        ax.scatter(player_positions[:, 0], player_positions[:, 1], color=player_colours, s=marker_size, edgecolors='white')

        # Circle size and offset calculations
        circle_radius = np.sqrt(marker_size / np.pi) * (PITCH_LENGTH / fig.dpi / 6)
        
        # Determine side to offset based on attack direction
        direction_multiplier = 1 if attack_direction.lower() == "right" else -1
        line_defender = back_player[0] + (direction_multiplier * circle_radius)
        line_attacker = forward_player[0] + (direction_multiplier * circle_radius)

        # Draw tangent line
        ax.plot([line_defender, line_defender], [0, PITCH_WIDTH], color='blue', linewidth=1)
        ax.plot([line_attacker, line_attacker], [0, PITCH_WIDTH], color=attacker_colour, linewidth=1)

        ax.scatter(forward_player[0], forward_player[1], color=attacker_colour, s=marker_size, edgecolors='white')

        ax.scatter(back_player[0], back_player[1], color='blue', s=marker_size, edgecolors='white')


    # Plot transformed referees if available
    if referees is not None:
        ax.scatter(referees[:, 0], referees[:, 1], color=ref_colour, s=marker_size, edgecolors='white', label='Referees')

    # Plot transformed goalkeepers if available
    if goalkeepers is not None:
        ax.scatter(goalkeepers[:, 0], goalkeepers[:, 1], color=goalkeeper_colour, s=marker_size, edgecolors='white', label='Goalkeepers')

    # Plot transformed footballs if available
    if footballs is not None:
        ax.scatter(footballs[:, 0], footballs[:, 1], color=football_colour, s=marker_size, edgecolors='white', label='Football')

    plt.show()

if __name__ == "__main__":
    pitch_display(players=None, referees=None, goalkeepers=None, footballs=None, transformed_points=None)