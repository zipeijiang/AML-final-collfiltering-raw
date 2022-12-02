import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from sklearn.neighbors import NearestNeighbors
#import data_preprocess

class Classifier:
    def __init__(self, playlists, songs, sparse_matrix, metric='cosine', n_neighbors=30, load_exists=False): # may change metric and n
        self.playlists = playlists
        self.songs = songs
        self.sparse_matrix = sparse_matrix
        
        self.model = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric)
        self.build(load_exists)
    
    def build(self, load_exists):
        if load_exists:
            self.load_model()
        else:
            self.train_model()
        
    def train_model(self):
        self.model.fit(self.sparse_matrix)
        pickle.dump(self.model, open(f'C:/Users/14809/Desktop/W4995AML/Final-Project/lib/NNmodel', 'wb'))
        
    def load_model(self):
        self.model = pickle.load(open(f"C:/Users/14809/Desktop/W4995AML/Final-Project/lib/NNmodel", "rb"))
    
    def predict_by_neighbors(self, neighbor_songs, known_songs, n_to_predict):
        scores = dict()
        for i, songs in enumerate(neighbor_songs):
            for song in songs:
                if song not in scores:
                    if song not in known_songs:
                        scores[song] = 1/(i+1) # may change weight
                else:
                    scores[song] += 1/(i+1)
        sorted_scores = sorted(scores, key=scores.get, reverse=True)[:n_to_predict]
        return sorted_scores
    
    def predict(self, X, n_to_predict):
        x_pid, x_songs = X['pid'], list(X['tracks'])
        x_sparse = dok_matrix((1,len(self.songs)), dtype=np.float32)
        x_sparse[0, x_songs] = 1
        x_vec = x_sparse.tocsr()
        
        x_neighbors = self.model.kneighbors(x_vec, return_distance=False)[0][1:]
        neighbor_songs = [self.playlists.loc[i]['tracks'] for i in x_neighbors]
        prediction = self.predict_by_neighbors(neighbor_songs, x_songs, n_to_predict)
        #print(prediction)        
        return prediction
