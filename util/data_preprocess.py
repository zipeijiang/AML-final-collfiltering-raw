import os
import json
import pickle
import numpy as np
import pandas as pd
import tqdm
from scipy.sparse import dok_matrix

FILES = os.listdir("C:/Users/14809/Desktop/W4995AML/Final-Project/data")
FILES.sort(key = lambda x: int(x.split('.')[2].split('-')[0]))

def load_data_to_df(num_files):
    
    track_uris = dict() # store all unique uris
    playlists_raw = [] # store uris and other info of each 
    tracks_raw = []
    
    files_to_read = FILES[:num_files]
    for file in tqdm.tqdm(files_to_read):
        data = json.load(open("C:/Users/14809/Desktop/W4995AML/Final-Project/data/"+file))
        playlists = data['playlists']  # 1k playlists
                
        for playlist in playlists:  
            tracks = playlist['tracks']
            for track in tracks:
                track_uri = track['track_uri'] # uri uniquely identifies a track
                if track_uri not in track_uris:
                    index = len(track_uris)
                    track_uris[track_uri] = index
                    tracks_raw.append(track)
                    #print(track_uris[track_uri] )
                track['track_uri'] = track_uris[track_uri] 
                
            playlist['tracks'] = [track['track_uri'] for track in tracks] # only want the uri
            playlists_raw.append(playlist)
        
    df_playlists = pd.DataFrame(playlists_raw)
    df_playlists.set_index('pid')
        
    df_tracks = pd.DataFrame(tracks_raw)
    df_tracks.set_index('track_uri')

    return df_playlists, df_tracks

def create_sparse_matrix(playlists, tracks):
    tids = list(tracks['track_uri'])
    pids = list(playlists['pid'])
    #tid_to_index = dict()
    #for i, tid in enumerate(tids):
    #    tid_to_index[tid] = i
    
    m, n = len(pids), len(tids)
    sparse_matrix = dok_matrix((m,n), dtype=np.float32)
    for i in tqdm.tqdm(range(m)):
        pid = pids[i]
        tids = playlists.loc[pid]["tracks"]
        #tids = [tid_to_index[tid] for tid in tids]
        sparse_matrix[pid, tids] = 1 

    return sparse_matrix.tocsr()#, tid_to_index

def update_tid_to_index(tracks, tid_to_index):
    tracks['numerical_uri'] = tracks.apply(lambda row: tid_to_index[row['track_uri']], axis=1)
    tracks.set_index('numerical_uri')
    return tracks

def preprocess_and_save_data(num_files):
    df_playlists, df_tracks = load_data_to_df(num_files)
    sparse_matrix = create_sparse_matrix(df_playlists, df_tracks)
    
    #df_tracks  = update_tid_to_index(df_tracks, tid_to_index)
    
    df_playlists.to_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/playlists.pkl")
    df_tracks.to_pickle("C:/Users/14809/Desktop/W4995AML/Final-Project/lib/tracks.pkl")
    pickle.dump(sparse_matrix, open(f"C:/Users/14809/Desktop/W4995AML/Final-Project/lib/sparse_matrix.pkl", "wb"))