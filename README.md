# Playlist Recommendation Engine

Digital audio streaming is on the rise, with a reported 61% of U.S. listeners streaming audio in 2021 ([MRC data](https://static.billboard.com/files/2021/09/U.S.-Music-360-2021-Draft-Report_Sneak-Preview_Sept-2021-2-1631178109.pdf)), and audio song streaming increasing globally by 25.6% between 2021 and 2022 ([Luminate 2022](https://luminatedata.com/reports/luminate-2022-u-s-year-end-report/?aliId=eyJpIjoieEN0ZFVqeklFU0RkYTZNeSIsInQiOiJhMFpUMXVSSHdEWlFNS1Rkcms2cDFnPT0ifQ%253D%253D)). Playlists play an increasingly important role in curating and organizing digital listening experiences, with a 2017 survey by [Nielsen Music](https://www.nielsen.com/insights/2017/music-360-2017-highlights/) finding that 75% of listeners who stream music online use playlists and over half create their own. As playlist creation and curation is often a time consuming process, users frequently turn to track recommendation systems to help them in constructing new playlists ([Inman et al., 2020](https://www.proquest.com/openview/fc6da445396fb9af102c5e8b4226db0d/1?pq-origsite=gscholar&cbl=30100)). Most existing track recommendation systems rely on listeners to provide seed tracks, and then utilize a variety of different approaches to recommend additional tracks in either a playlist-like listening session or as sequential track recommendations based on user feedback.

Our recommendation engine takes a different approach, allowing listeners to generate playlists based on a semantic string, such as the title of desired playlist, specific mood (happy, relaxed), atmosphere (tropical), or function (party music, focus) [this needs work]. Using a publicly available dataset of [existing playlists](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge), we combine a semantic similarity vector model with a matrix factorization model to allow users to quickly and easily generate playlists to fit any occasion.



# Data Processing

The [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) (or MPD for short) is a publicly available compilation of 1 million playlists created by Spotify users. It consists of 1,000 json files, each containing 1,000 playlists. A playlist consists of metadata (e.g. a unique playlist id, name, number of followers, number of tracks, etc.) and a list of tracks with their respective metadata (e.g. name, artist, album, etc.). Given that a track usually appears in multiple playlists, we created a SQL database to avoid storing redundant metadata and to improve processing times when training and subsequently using our models. The original MPD requires 5.8 GB of storage when compressed in a zip folder, and is 33.5 GB when uncompressed, whereas our SQL database uses only 1.6 GB of storage, making it a more efficient means of storing and interacting with the dataset. This is our schema:

```markdown
| PLAYLISTS           | PAIRINGS | TRACKS      |
|---------------------|----------|-------------|
| PID                 | PID      | TID         |
| Name                | TID      | Track Name  |
| Number of Followers | pos      | Artist Name |
| Number of Albums    |          | Album Name  |
| Number of Artists   |          | Track URI   |
| Number of Tracks    |          | Album URI   |
| Number of Edits     |          | Artist URI  |
| Collaborative       |          | Duration    |
| Modified At         |          |             |
| Duration            |          |             |
```

# Overview of Pipeline

![Pipeline](https://github.com/dymiyata/erdos2023_million_playlist_challenge/assets/142643458/ebfa8df3-10d3-4527-9f3f-c856caec74d3)



# Semantic Search

Our recommendation engine begins by generating a seed playlist from a text input by the user. This seed playlist is then filtered by the user, and fed into the matrix factorization model to generate a longer playlist of similar songs. We construct the seed playlist using a semantic search algorithm, which allows flexibility in the way a user interacts with the engine. Specifically the seed playlist is constructed using the following vector search algorithm:

First we assign a vector to each playlist in the database by passing its title to the pre-trained NLP model [SBERT](https://www.sbert.net/) (using model='all-MiniLM-L6-v2' ). This assignment of a vector to each playlist is such that titles with similar meaning are closer together with respect to the cosine similarity metric. The user's input is then vectorized using the same model. From here we wish to find the playlist vectors which are nearest to the user's input vector. As our database is large, we use the approximate nearest neighbors algorithm [ANNOY](https://github.com/spotify/annoy) to quickly find the playlists whose titles are most semantically similar to the user's input. With these playlists in hand we rank their corresponding pool of songs by how many nearby playlists it occurs in. The seed playlist is then created as the top n tracks from the aforementioned pool of songs. 



# Matrix Factorization Model

As the MPD contains no song metadata such as genre, we use a form of collaborative filtering called matrix factorization (MF) to recommend songs similar to those in the seed. The defining feature of MF algorithms is that they assemble data into a large matrix which is then approximated by a product of two smaller matrices. In our case we use our data set to assemble a matrix, $R$, whose rows and columns correspond to playlists and tracks respectively, and whose entries consist of a 1 if the track is in the playlist and a zero otherwise. That is, 

$$
R_{\text{playlist},\text{track}} = 
\begin{cases} 
1 & \text{track is in playlist}\\
0 & \text{otherwise}.  
\end{cases}
$$

Then if we have matrices $P$ and $Q$ so that $PQ^T \approx R$ we can treat the seed playlist as an "unfinished row" in the matrix $R$, and use the matrices $P$ and $Q$ to complete it.


We found the matrices $P$ and $Q$ by implementing a MF algorithm called Funk SVD from scratch. This a machine learning algorithm whose prominent hyper-parameter is the number of "latent features", which we will denote by $f$. This hyper-parameter determines the size of $P$ and $Q$. That is, as $R$ is a $|\text{playlists}| \times |\text{tracks}|$ matrix, the matrices $P$ and $Q$ will have dimension $|\text{playlists}| \times f$ and $f \times |\text{tracks}|$ respectively. The loss function for this algorithm is given by the squared error

$$
SE = \sum_{R_{i,j}= 1} (1- P_iQ_j^T)^2.
$$

Beginning with random values for $P$ and $Q$ we use gradient descent with an $L_2$ regularization term to update the values.

We split our data into a 70/15/15 train-test-validation split. We used cross validation to select the hyper-parameter $f = 20$. The value of the error function on the test set 0.03512.

**Note:** Due to time constraints, the reported MSE was computed only on 10,000 out of the 150,000 playlists available in the test set. The computation of the $P$ matrix corresponding to the full test set was taking too much time.


# Cosine Similarity Filter

The output of the MF model is 1000 recommended songs and a $f$-dimensional vectors for each song. To refine these results we compute the cosine similarity of each of these songs with our seed tracks. We then select the 50 tracks from the 1000 initial tracks with the best cosine similarity score.  
 


# GUI



# Future directions
We plan to deploy this model on an AWS cloud server so that we can collect information from user interaction and feedback with the recommendation engine to further improve our model. Additional layers can also be added to improve recommendations, including adding weights for tracks appearing in playlists that have a high number of followers and incorporating the [MuSe (Musical Sentiment) database](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset) to further refine recommendations for specific moods or emotions.
