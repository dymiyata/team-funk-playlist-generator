import pandas as pd
import numpy as np
import time

from tkinter import *
from tkinter import ttk
from functools import partial

from sqlalchemy import create_engine
from sqlalchemy import text

from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer



"""
To run this script the following must be done:
    sql database built and saved in MPD_sql.db in current working directory
    annoy database build and saved in playlist_vectors.ann in current working directory
    Q matrix trained and saved in Q_train_small_order in current working directory
"""





def list_to_dict(new_list):
    # We only care about unique elements
    # Also, np.unique returns the sorted list
    new_list = np.unique(new_list)
    
    elt_to_idx = {}
    #idx_to_elt = {}
    
    idx=0
    for elt in new_list:
        #idx_to_elt[idx] = elt
        elt_to_idx[elt] = idx
        
        idx+=1
    
    return elt_to_idx #, idx_to_elt



def new_user_vec(tid_index_list, Q, llambda):
    Y = Q[tid_index_list,:]
    f = np.shape(Q)[1]
    d = len(tid_index_list)
    vec = np.linalg.inv(np.transpose(Y) @ Y + llambda * np.identity(f)) @ np.transpose(Y) @ np.ones((d,1))
    return np.transpose(vec)

def refine_cos_sim(candidates, tid_index_list, Q):
    score_dict = {}
    for i in candidates:
        score_dict[i] = max([cos_sim(Q[i], Q[j]) for j in tid_index_list])
    sorted_scores = sorted(score_dict, key=score_dict.get)
    return sorted_scores 


def cos_sim(A, B):
    return A @ B / (np.linalg.norm(A) * np.linalg.norm(B))


def get_rec_indices(playlist_tids, Q_current, llambda, playlist_length, tid_ind_dict):
    num_songs = np.shape(Q_current)[0]

    tid_indices = [tid_ind_dict[tid] for tid in playlist_tids]

    # Find playlist vector that minimizes cost
    x = new_user_vec(tid_indices, Q_current, llambda)

    # Find candidate songs
    num_candidates = 1000
    pred_x = np.square((x @ np.transpose(Q_current)).reshape(num_songs) - 1)
    candidate_tids = np.argpartition(pred_x, num_candidates)[0:num_candidates]

    # Create list of recommendations
    num_recs_per = playlist_length // len(playlist_tids)
    rec_array = np.zeros((len(playlist_tids), num_recs_per)) - 1

    # For each tid in playlist_tids, find num_recs_per recommendations based on cosine similarity
    # for i in range(len(tid_indices)):
    #     current_ind = tid_indices[i]
    #     sorted_scores = refine_cos_sim(candidate_tids, [current_ind], Q_current)[::-1]

    #     count = 0
    #     cand_index = 0
    #     while count < num_recs_per:
    #         cand_tid = sorted_scores[cand_index]
    #         if (cand_tid not in rec_array):
    #             rec_array[i, count] = cand_tid
    #             count += 1
    #         cand_index += 1

    rec_vec = rec_array.flatten()

    rec_vec = refine_cos_sim(candidate_tids, tid_indices, Q_current)[::-1]

    np.random.shuffle(rec_vec)
    return rec_vec[0:10]




def get_recs(playlist_tids, Q_current, llambda, playlist_length):
    num_songs = np.shape(Q_current)[0]

    # Find playlist vector that minimizes cost
    x = new_user_vec(playlist_tids, Q_current, llambda)

    # Find candidate songs
    num_candidates = 10000
    pred_x = np.square((x @ np.transpose(Q_current)).reshape(num_songs) - 1)
    candidate_tids = np.argpartition(pred_x, num_candidates)[0:num_candidates]

    # Create list of recommendations
    num_recs_per = playlist_length // len(playlist_tids)
    rec_array = np.zeros((len(playlist_tids), num_recs_per)) - 1



    # For each tid in playlist_tids, find num_recs_per recommendations based on cosine similarity
    for i in range(len(playlist_tids)):
        current_tid = playlist_tids[i]
        sorted_scores = refine_cos_sim(candidate_tids, [current_tid], Q_current)[::-1]

        count = 0
        cand_index = 0
        while count < num_recs_per:
            cand_tid = sorted_scores[cand_index]
            if (cand_tid not in rec_array):
                rec_array[i, count] = cand_tid
                count += 1
            cand_index += 1

    rec_vec = rec_array.flatten()
    np.random.shuffle(rec_vec)
    return rec_vec



#function takes in a string and out puts pid's in a list
#the size of the output list is variable
#if there are many results which match the query exactly then it will increase the size of the list
def search_playlist_titles(query):
    #generate search vector
    v_search = model.encode(query.lower())
    
    pid_nns, dist = t.get_nns_by_vector(v_search, 10000, search_k = 100000, include_distances = True )
    
    #find first nonzero distance
    i = 0 
    while dist[i] == 0.0:
        i = i +1

    if i < 25:
        return pid_nns[:25]
    else:
        return pid_nns[:i]


# Clears the item in a treeview widget
def clear_tree(tree):
    for item in tree.get_children():
        tree.delete(item)

def search_click(tree, trackEntry, artistEntry):
    clear_tree(tree)

    trackString = trackEntry.get()
    artistString = artistEntry.get()
    query = f"SELECT * FROM tracks WHERE artist_name LIKE \'%{artistString}%\' AND track_name LIKE \'%{trackString}%\'"
    track_db = pd.read_sql_query(query, conn)

    for index, row in track_db.head(20).iterrows():
        tree.insert(
            parent='', 
            index = 'end', 
            iid=row['tid'], text="",
            values = (
                row['track_name'], 
                row['artist_name'], 
                row['album_name']
            )
        )

def add_song(tree):
    # Gets iids (which are tids) of selected items
    curItems = tree.selection()     
    
    # A list of dictionaries, one per item
    itemList = [(tree.item(i), i) for i in curItems]

    # Add each item to the seedTree
    for selectDict, i in itemList:
        seedTree.insert(
            parent='', 
            index='end', 
            iid=i, 
            text="", 
            values = (
                selectDict['values'][0], 
                selectDict['values'][1], 
                selectDict['values'][2]
            )
    )
    tree.master.destroy()

def search_song():
    topSearch = Toplevel(root, height = 15)
    topSearch.title("Find Specific Song")
    topSearch.geometry("1000x600")

    upperFrame = Frame(topSearch)
    upperFrame.pack(pady=20)

    trackFrame = Frame(topSearch)
    trackFrame.pack(in_=upperFrame, side=LEFT)

    artistFrame = Frame(topSearch)
    artistFrame.pack(in_=upperFrame, side=RIGHT)

    trackNameLabel = Label(topSearch, text="Enter Song Name")
    trackNameLabel.pack(in_=trackFrame, side=TOP)

    getTrackName = Entry(topSearch)
    getTrackName.pack(in_=trackFrame, side=BOTTOM, padx = 20, pady = 20)

    artistNameLabel = Label(topSearch, text="Enter Artist Name")
    artistNameLabel.pack(in_=artistFrame, side=TOP)

    getArtistName = Entry(topSearch)
    getArtistName.pack(in_=artistFrame, side=BOTTOM, padx = 20, pady = 20)

    resultsTree = ttk.Treeview(topSearch)

    actionWithArgs = partial(search_click, resultsTree, getTrackName, getArtistName)
    searchButton = Button(topSearch, 
                          text="Search Song",
                          command = actionWithArgs)
    searchButton.pack(padx=10, pady = 10)

    resultsTree['columns'] = ("Track Name", "Artist Name", "Album")
    resultsTree.column("#0", width = 0)
    resultsTree.column("Track Name", anchor=W, width = 250, minwidth = 25)
    resultsTree.column("Artist Name", anchor=W, width = 250, minwidth = 25)
    resultsTree.column("Album", anchor=W, width = 250, minwidth = 25)
    #resultsTree.heading("#0", text="", anchor=W)
    resultsTree.heading("Track Name", text="Track Name", anchor=W)
    resultsTree.heading("Artist Name", text="Artist Name", anchor = W)
    resultsTree.heading("Album", text="Album", anchor = W)
    resultsTree.pack()

    addButton = Button(topSearch, text="Add Selection to Seeds", command=partial(add_song, resultsTree))
    addButton.pack(padx=10, pady=10)

def phrase_click(tree, entry):
    global did_search
    did_search = True
    clear_tree(tree)

    search = entry.get()
    pid_nns = search_playlist_titles(search)
    pid_nns_str = ",".join(map(str, pid_nns))

    # Query the sql database and rank songs in the playlists by number of appearances
    query = f"""SELECT tid, COUNT(*) as frequency 
        FROM pairings 
        WHERE pid IN ({pid_nns_str}) 
        GROUP BY tid 
        ORDER BY frequency DESC"""
    global ordered_tids 
    ordered_tids = conn.execute(text(query)).fetchall()
    global rec_index 
    rec_index = 0

    # Get top 10 songs
    top_10 = [row[0] for row in ordered_tids[:10]]
    top_10_str = ",".join(f"'{tid}'" for tid in top_10)
    query = f"""SELECT * FROM tracks
        WHERE tid IN ({top_10_str})"""
    top_10 = conn.execute(text(query)).fetchall()

    for track in top_10:
        tree.insert(
            parent='', 
            index='end', 
            iid=track[0], 
            text="", 
            values = (
                track[1], 
                track[2], 
                track[3]
            )
    )
        
    for row in top_10:
        tid = row[0]
        print(Q_trained[tid_to_idx[tid]])

        
def regen_click(tree):
    if not did_search:
        return

    clear_tree(tree)

    global rec_index 

    rec_index += 1

    # Get top 10 songs
    top_10 = [row[0] for row in ordered_tids[rec_index*10:rec_index*10+10]]
    top_10_str = ",".join(f"'{tid}'" for tid in top_10)
    query = f"""SELECT * FROM tracks
        WHERE tid IN ({top_10_str})"""
    top_10 = conn.execute(text(query)).fetchall()

    for track in top_10:
        tree.insert(
            parent='', 
            index='end', 
            iid=track[0], 
            text="", 
            values = (
                track[1], 
                track[2], 
                track[3]
            )
    )

def search_phrase():
    global did_search
    did_search = False
    topSearch = Toplevel(root)
    topSearch.title("Generate Recommended Seeds from Phrase")
    topSearch.geometry("1000x600")

    phraseLabel = Label(topSearch, text="Enter Phrase")
    phraseLabel.pack(padx=10, pady=10)

    getPhrase = Entry(topSearch)
    getPhrase.pack(padx = 20, pady = 20)

    resultsTree = ttk.Treeview(topSearch, height = 15)

    searchFrame = Frame(topSearch)
    searchFrame.pack(pady=20)

    actionWithArgs = partial(phrase_click, resultsTree, getPhrase)
    searchButton = Button(topSearch, 
                          text="Generate Recommended Seeds",
                          command = actionWithArgs)
    searchButton.pack(in_=searchFrame, side=LEFT, padx=10, pady = 10)

    regenButton = Button(topSearch, 
                          text="Regenerate",
                          command = partial(regen_click, resultsTree))
    regenButton.pack(in_=searchFrame, side=LEFT, padx=10, pady = 10)

    resultsTree['columns'] = ("Track Name", "Artist Name", "Album")
    resultsTree.column("#0", width = 0)
    resultsTree.column("Track Name", anchor=W, width = 250, minwidth = 25)
    resultsTree.column("Artist Name", anchor=W, width = 250, minwidth = 25)
    resultsTree.column("Album", anchor=W, width = 250, minwidth = 25)
    #resultsTree.heading("#0", text="", anchor=W)
    resultsTree.heading("Track Name", text="Track Name", anchor=W)
    resultsTree.heading("Artist Name", text="Artist Name", anchor = W)
    resultsTree.heading("Album", text="Album", anchor = W)
    resultsTree.pack()

    buttonFrame = Frame(topSearch)
    buttonFrame.pack(pady=10)

    addButton = Button(topSearch, text="Add Selection to Seeds", command=partial(add_song, resultsTree))
    addButton.pack(in_=buttonFrame, side=LEFT, padx=10, pady=10)
    allButton = Button(topSearch, text="Select All", command=partial(select_all, resultsTree))
    allButton.pack(in_=buttonFrame, side=RIGHT, padx=10, pady=10)



def select_all(tree):
    for item in tree.get_children():
        tree.selection_add(item)

def create_playlist():
    seed_tids = [int(iid) for iid in seedTree.get_children()]
    # for i in seed_tids:
    #     print(Q_trained[tid_to_idx[i], :])

    rec_indices = get_rec_indices(seed_tids, Q_trained, llambda, 50, tid_to_idx)
    rec_tids = [tid_train[int(i)]  for i in rec_indices]

    rec_tid_str = ",".join(f"'{tid}'" for tid in rec_tids)
    query = f"""SELECT * FROM tracks
        WHERE tid IN ({rec_tid_str})"""
    recs = conn.execute(text(query)).fetchall()

    for track in recs:
        print(str(track[1]) + ": " + str(track[2]))

    # for track in recs:
    #     tree.insert(
    #         parent='', 
    #         index='end', 
    #         iid=track[0], 
    #         text="", 
    #         values = (
    #             track[1], 
    #             track[2], 
    #             track[3]
    #         )
    # )
    return

def create_playlist_ordered():
    seed_tids = [int(iid) for iid in seedTree.get_children()]
    seed_tids_str = ",".join(f"\'{tid}\'" for tid in seed_tids)
    query = f"""SELECT track_uri FROM tracks
        WHERE tid in ({seed_tids_str})"""
    seed_uris = conn.execute(text(query)).fetchall()
    seed_ordered_ind = [track_dict[uri[0]] for uri in seed_uris]
    # for i in seed_tids:
    #     print(Q_trained[tid_to_idx[i], :])

    rec_indices = get_recs(seed_ordered_ind, Q_trained, llambda, 50)
    rec_tids = [int(i) for i in rec_indices if i < 681805]
    rec_uris = [reverse_track_dict[tid] for tid in rec_tids]

    rec_uri_str = ",".join(f"\'{uri}\'" for uri in rec_uris)
    query = f"""SELECT * FROM tracks
        WHERE track_uri IN ({rec_uri_str})"""
    recs = conn.execute(text(query)).fetchall()

    resultTop = Toplevel(root)
    resultTop.title("Your Playlist")
    resultTop.geometry("800x800")

    resultsTree = ttk.Treeview(resultTop, height = 30)

    searchFrame = Frame(resultTop)
    searchFrame.pack(pady=20)

    resultsTree['columns'] = ("Track Name", "Artist Name", "Album")
    resultsTree.column("#0", width = 0)
    resultsTree.column("Track Name", anchor=W, width = 250, minwidth = 25)
    resultsTree.column("Artist Name", anchor=W, width = 250, minwidth = 25)
    resultsTree.column("Album", anchor=W, width = 250, minwidth = 25)
    #resultsTree.heading("#0", text="", anchor=W)
    resultsTree.heading("Track Name", text="Track Name", anchor=W)
    resultsTree.heading("Artist Name", text="Artist Name", anchor = W)
    resultsTree.heading("Album", text="Album", anchor = W)
    resultsTree.pack()

    removeButton = Button(resultTop, text="Remove Selected Tracks", command = partial(delete_tree_item, resultsTree))
    removeButton.pack(padx=10, pady=10)

    saveButton = Button(resultTop, text="Save as .csv")
    saveButton.pack(padx=10, pady=10)

    # for track in recs:
    #     print(track[1], track[2])

    for track in recs:
        resultsTree.insert(
            parent='', 
            index='end', 
            iid=track[0], 
            text="", 
            values = (
                track[1], 
                track[2], 
                track[3]
            )
    )
    return

def delete_tree_item(tree):
    # Gets iids (which are tids) of selected items
    curItems = tree.selection()     
    
    # A list of dictionaries, one per item
    for iid in curItems:
        tree.delete(iid)




# Load the annoy database
model = SentenceTransformer('all-MiniLM-L6-v2')
f = model.get_sentence_embedding_dimension()
t=AnnoyIndex(f, 'angular')
t.load('playlist2vec/playlist_vectors.ann')

# Connect to the sql database
engine = create_engine("sqlite:///MPD_sql.db")
conn = engine.connect()

# Load trained track matrix and list of tids for tracks
llambda = 10**(-3)
# Q_trained = np.load('Q_trained_500.npy')

Q_trained = np.load('Q_train_small_ordered.npy')
reverse_track_dict = np.load('ordered_reverse_track_dict.npy', allow_pickle='TRUE').item()
track_dict = np.load('ordered_track_dict.npy', allow_pickle='TRUE').item()
tid_train = np.load('tid_train.npy')
tid_to_idx = list_to_dict(tid_train)



# Define boolean which tracks if a playlist name search was made
did_search = False





############## Create GUI #################


root = Tk()
root.geometry("1000x750")


instructions = Label(root, text='Get recommended seed tracks from phrase\n or search for specific tracks', font=('Arial, 16'))
instructions.pack(pady=10)

topFrame = Frame(root)
topFrame.pack(padx=10, pady=20)



phraseButton = Button(root, text='Get Recommended Seeds from Phrase', command = search_phrase)
phraseButton.pack(in_=topFrame, side=LEFT, padx=10)

findSongButton = Button(root, text="Find Specific Track", command = search_song)
findSongButton.pack(in_=topFrame, side=RIGHT, padx=10)

seedTreeLabel = Label(root, text="Seeds")
seedTreeLabel.pack(pady=10)
seedTree = ttk.Treeview(root, height=15)
seedTree['columns'] = ("Track Name", "Artist Name", "Album")
seedTree.column("#0", width = 0)
seedTree.column("Track Name", anchor=W, width = 250, minwidth = 25)
seedTree.column("Artist Name", anchor=W, width = 250, minwidth = 25)
seedTree.column("Album", anchor=W, width = 250, minwidth = 25)
#seedTree.heading("#0", text="", anchor=W)
seedTree.heading("Track Name", text="Track Name", anchor=W)
seedTree.heading("Artist Name", text="Artist Name", anchor = W)
seedTree.heading("Album", text="Album", anchor = W)
seedTree.pack()

buttonFrame = Frame(root)
buttonFrame.pack(pady=10)

removeButton = Button(root, text="Remove Selected\n Seed Track", command = partial(delete_tree_item, seedTree))
removeButton.pack(in_=buttonFrame, side=RIGHT, padx=10, pady=10)

completeButton = Button(root, text="Generate Playlist\n From Seeds", command = create_playlist_ordered)
completeButton.pack(in_=buttonFrame, side=LEFT, padx=10, pady=10)


root.mainloop()