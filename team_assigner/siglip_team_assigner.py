from sklearn.cluster import KMeans
import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
from tqdm import tqdm
import supervision as sv
from more_itertools import chunked
import umap.umap_ as umap

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class TeamAssignerSigLip:
    def __init__(self):
        self.team_colours = {0: [0, 0, 255]}
        self.player_team_dict = {}

    def get_clustring_model(self, image, n_clusters=2):
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_colour(self, frame, bbox):    
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image_center_box = image[image.shape[0]//4:(2*image.shape[0])//4, image.shape[1]//3:(3*image.shape[1])//5]

        if image_center_box.size != 0:
            kmean = self.get_clustring_model(image_center_box, n_clusters=5)

            # get the cluster labels for each pixel
            labels = kmean.labels_

            # get player cluster 
            player_colour = kmean.cluster_centers_[np.argmax([sum(labels==i) for i in range(5)])]
            # player_colour = kmean.cluster_centers_[np.argsort([sum(labels==i) for i in range(5)])]
            # take avrage of the two highest clusters
            # player_colour = np.mean(player_colour[:2], axis=0)
            return player_colour

        else:
            player_colour = None
            return player_colour


    def assign_team_colour(self):
        self.team_colours[1] = [0, 0, 255] # red
        self.team_colours[2] = [0, 255, 0] # green


    def assign_players_to_teams(self, frame_num, player_id, player_idx):
        """
        Assigns a player to a team based on Siglip embeddings and clustering.

        Args:
            frame_num: The frame number where the player was detected.
            player_id: Unique identifier for the player.
            player_idx: Index of the player in the frame.

        Returns:
            team_id: The assigned team ID for the player.
        """
        # Retrieve the embedding index from the frame-player map
        embedding_index = self.frame_player_map.get((frame_num, player_idx))

        if embedding_index is not None:
            # Determine team ID from clustering results
            team_id = self.clusters[embedding_index] + 1  # Convert cluster ID (0-based) to team ID (1-based)
            self.player_team_dict[player_id] = team_id
            return team_id
        else:
            # Return 0 if no embedding is found
            return 0

    
class Siglip():
    def __init__(self):
        self.EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH, attn_implementation="sdpa").to(DEVICE).eval()
        self.EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH,)
        self.REDUCER = umap.UMAP(n_components=3)
        self.CLUSTERING_MODEL = KMeans(n_clusters=2)
        self.clusters = {}
        self.frame_player_map = {}

    def crop_players(self, frames, detections):
        PLAYER_ID = 2  # Replace with the correct class ID for players
        crops = []
        embedding_index = 0
        self.frame_player_map = {}

        for frame_idx, (frame, detection) in enumerate(zip(frames, detections)):
            # Filter detections by the class ID (e.g., PLAYER_ID)
            player_detections = detection[detection.cls == PLAYER_ID]
            
            # Crop images for each player detected in the frame
            players_crops = [sv.crop_image(frame, xyxy.cpu().numpy()) for xyxy in player_detections.xyxy]

            # Map each player in each frame to an embedding index
            for player_idx, _ in enumerate(players_crops):
                self.frame_player_map[(frame_idx, player_idx)] = embedding_index
                embedding_index += 1
            
            # Collect all crops
            crops += players_crops

        return crops
 
    def embedding_players(self, crops, batch_size=64):

        batches = chunked(crops, batch_size)
        data = []
        for batch in tqdm(batches, desc='embedding extraction'):
            embeddings = self.embedding_(batch)
            data.append(embeddings)

        data = np.concatenate(data)
        return data
    
    @torch.no_grad()
    def embedding_(self, player):
        inputs = self.EMBEDDINGS_PROCESSOR(images=player, return_tensors="pt").to(DEVICE)
        outputs = self.EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        return embeddings
    
    def fit(self, data):
        projections = self.REDUCER.fit_transform(data)
        self.CLUSTERING_MODEL.fit(projections)

    def predict(self, data, is_umap=True):
        if is_umap:
            projections = self.REDUCER.transform(data)
        else:
            projections = data

        cluster_model = self.CLUSTERING_MODEL.predict(projections)
        for i in range(len(data)):
            self.clusters[i] = cluster_model[i]
        return cluster_model
    
    def pipeline(self, frames, detections):
        crops = self.crop_players(frames, detections)
        data = self.embedding_players(crops)
        self.fit(data)
        return self.predict(data), self.frame_player_map