import os
import json
import pickle
import numpy as np
import pandas as pd
import tqdm
import random
from util import data_preprocess, classifier

# github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
def getPlaylistTracks(playlist, songs):
    return [songs.loc[x] for x in playlist["tracks"]]

# github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
def getTrackandArtist(trackURI, songs):
    song = songs.loc[trackURI]
    return (song["track_name"], song["artist_name"])

class Main:
    def __init__(self, load_exists=True, sample_size=100):
        self.initialize(load_exists, sample_size)
        self.build_classifier()
        
    def initialize(self, load_exists, sample_size):
        if not load_exists:
            data_preprocess.preprocess_and_save_data(sample_size)
        self.playlists = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/playlists.pkl")
        self.songs = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/tracks.pkl")
        self.playlistSparse = pd.read_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/sparse_matrix.pkl")
        
    def build_classifier(self):
        self.classifier = classifier.Classifier(
            self.playlists, self.songs, self.playlistSparse,
            metric='cosine', n_neighbors=30, load_exists=False) # TBC
    
    def predict(self, playlist, n_to_predict):
        return self.classifier.predict(playlist, n_to_predict)
    
    # github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
    def getRandomPlaylist(self): 
        return self.playlists.iloc[random.randint(0,len(self.playlists) - 1)]
    
    # github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
    def obscurePlaylist(self, playlist, obscurity): 
        """
        Obscure a portion of a playlist's songs for testing
        """
        k = int(len(playlist['tracks']) * obscurity)
        indices = random.sample(range(len(playlist['tracks'])), k)
        obscured = [playlist['tracks'][i] for i in indices]
        tracks = [i for i in playlist['tracks'] + obscured if i not in playlist['tracks'] or i not in obscured]
        return tracks, obscured

    # github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
    def evalAccuracy(self, numPlaylists, percentToObscure=0.5): 
        """
        Obscures a percentage of songs
        Iterates and sees how many reccomendations match the missing songs
        """
        print()
        print(f"Selecting {numPlaylists} playlists to test and obscuring {int(percentToObscure * 100)}% of songs")

        def getAcc(pToObscure):
            playlist = self.getRandomPlaylist()

            keptTracks, obscured = self.obscurePlaylist(playlist, pToObscure)
            playlistSub = playlist.copy()
            obscured = set(obscured)
            playlistSub['tracks'] = keptTracks

            predictions = self.predict(playlistSub, 
                500)

            overlap = [value for value in predictions if value in obscured]

            return len(overlap)/len(obscured)
        
        accuracies = [getAcc(percentToObscure) for _ in tqdm.tqdm(range(numPlaylists))]
        avgAcc = round(sum(accuracies) / len(accuracies), 4) * 100
        print(f"we predicted {avgAcc}% of obscured songs")
    
    # github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
    def displayRandomPrediction(self):
        playlist = self.getRandomPlaylist()
        while len(playlist["tracks"]) < 10:
            playlist = self.getRandomPlaylist()

        predictions = self.predict(playlist=playlist,
            n_to_predict=50)


        playlistName = playlist["name"]
        playlist = [getTrackandArtist(trackURI, self.songs) for trackURI in playlist["tracks"]]
        predictions = [getTrackandArtist(trackURI, self.songs) for trackURI in predictions]
        return {
            "name": playlistName,
            "playlist": playlist,
            "predictions": predictions
        }
    
    # github.com/wsmiles000/CS109a-Spotify-Recommendation#base-line-model-recommending-tracks-by-popularity
    def createRandomPredictionsDF(self, numInstances):
        print(f"Generating {numInstances} data points")
        data = [self.displayRandomPrediction() for _ in tqdm.tqdm(range(numInstances))]
        df = pd.DataFrame(data)
        df.to_csv("C:/Users/14809/Desktop/W4995AML/Final-Project/predictionData.csv")


if __name__ == "__main__":
    # Init class
    explorer = Main(load_exists=True, sample_size=100)

    #Run tests on NNC
    explorer.evalAccuracy(30)

    # Generate prediction CSV
    explorer.createRandomPredictionsDF(100)
