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


![Pipeline)](https://github.com/dymiyata/team-funk-playlist-generator/assets/142643458/58037c82-31d0-4015-aab0-c9437beb3329)




# Semantic Search

Our recommendation engine begins by generating a seed playlist from a text input by the user. This seed playlist is then filtered by the user, and fed into the matrix factorization model to generate a longer playlist of similar songs. We construct the seed playlist using a semantic search algorithm, which allows flexibility in the way a user interacts with the engine. Specifically the seed playlist is constructed using the following vector search algorithm:

First we assign a vector to each playlist in the database by passing its title to the pre-trained NLP model [SBERT](https://www.sbert.net/) (using model='all-MiniLM-L6-v2' ). This assignment of a vector to each playlist is such that titles with similar meaning are closer together with respect to the cosine similarity metric. The user's input is then vectorized using the same model. From here we wish to find the playlist vectors which are nearest to the user's input vector. As our database is large, we use the approximate nearest neighbors algorithm [ANNOY](https://github.com/spotify/annoy) to quickly find the playlists whose titles are most semantically similar to the user's input. With these playlists in hand we rank their corresponding pool of songs by how many nearby playlists it occurs in. The seed playlist is then created as the top n tracks from the aforementioned pool of songs. 



# Matrix Factorization Model (Funk SVD)

As the MPD contains no song metadata such as genre, we use a form of collaborative filtering called matrix factorization (MF) to recommend songs similar to those in the seed. The defining feature of MF algorithms is that they assemble data into a large matrix which is then approximated by a product of two smaller matrices. In our case we use our data set to assemble a matrix, $R$, whose rows and columns correspond to playlists and tracks respectively, and whose entries consist of a 1 if the track is in the playlist and a zero otherwise. That is, 

$$
R_{\text{playlist},\text{track}} = 
\begin{cases} 
1 & \text{track is in playlist}\\
0 & \text{otherwise}.  
\end{cases}
$$

Just because a track is not in a playlist, doesn't mean that it shouldn't belong there. For instance, the creator of the playlist may not be aware of the certain track that they would have added to their playlist had they heard the track. To this end, the key idea is to treat the zeros in $R$ as missing values that we wish to fill in with other values to get a final matrix $R_{pred}$.  If an entry of $R_{pred}$ is close to 1 then our model will recommend the corresponding track to add to the that playlist. 

To predict these missing values we wish to find smaller matrices $P$ (corresponding to playlists) and $Q$ (corresponding to tracks) so that the product $PQ^T$ matches the matrix $R$ on the entries equal to 1. Once we have such matrices $P$ and $Q$ we set:

$$
R_{pred} = PQ^T.
$$

We found $P$ and $Q$ by implementing a MF algorithm called Funk SVD from scratch which learns $P$ and $Q$ using gradient descent. The prominent hyper-parameter in Funk SVD is the number of "latent features", which we will denote by $f$. This hyper-parameter determines the size of $P$ and $Q$. That is, as $R$ is a $|\text{playlists}| \times |\text{tracks}|$ matrix, the matrices $P$ and $Q$ will have dimension $|\text{playlists}| \times f$ and $f \times |\text{tracks}|$ respectively.  The rows of $P$ are $f$-dimensional vectors, one for each playlists and the rows of $Q$ are $f$-dimensional vectors, one for each track. Since we want $PQ^T$ to match $R$ on the entries equal to 1, our loss function for this algorithm is given by the squared error

$$
SE = \sum_{R_{i,j}= 1} (1- P_iQ_j^T)^2.
$$

Beginning with random values for $P$ and $Q$ we use gradient descent with an $L_2$ regularization term to update the entries of $P$ and $Q$.  

Once $P$ and $Q$ have been learned, the next question is how do we recommend songs to a partial playlist $x$ that is not in our dataset? To do this, we treat the matrix $Q$ as fixed and find the vector $p_{x}$ that minimizes the squared error

$$
SE_{x} = \sum_{i} (1- p_x Q_i^T)^2
$$

where the sum is taken over indices $i$ corresponding to tracks in the partial playlist $x$.  Since $Q$ is fixed, $SE_{x}$ treated as a function depending on $p_x$ is convex. Thus, the vector $p_x$ that minimizes $SE_{x}$ can be computed directly.  Then, to recommend tracks to add to playlist $x$, we simply find the tracks with index $i$ where $p_xQ_i^T \approx 1$ 

## Choosing Values for Hyperparameters

We split our data into a 70/15/15 train-test-validation split in order to choose the values of our hyperparameters, most notably $f$. For each playlist $x$ in our validation set, we find $p_{x}$ as described above and use the sum of squared errors $\sum_{x} SE_{x}$ as the error on our validation set. We found that selecting $f = 20$ minimized this error on the validation set. 

To test how well our model performs we repeat this operation on our test set, but use Mean Squared Error (MSE) rather than SE. We found a MSE of 0.03512 on our test set.

**Note:** Due to time constraints, the reported MSE was computed only on 10,000 out of the 150,000 playlists available in the test set. 
<!--- The computation of the $P$ matrix corresponding to the full test set was taking too much time. --->


## Cosine Similarity Filter

The output of the MF model is 1000 recommended songs and a $f$-dimensional vectors for each song. To refine these results we then compute the cosine similarity of each of these songs with our seed tracks. Finally, we select the 50 tracks from the 1000 initial tracks with the best cosine similarity score.  
 


# Demonstration of Playlist Generation

In order to interact with our playlist generation engine, we build a simple graphical user interface. At first the user sees the window shown below:

<p align="center">
 <img src="https://github.com/dymiyata/team-funk-playlist-generator/assets/48339284/03e7fa13-a68a-47a8-a856-b0cdc51a7092" width="500" >
</p>

Since the MF portion of our engine requires a partial playlist, the user must first determine the seed tracks used to generate the full playlist. To do so, the user has two options. They can search for specific tracks if they have some in mind by clicking the button labeled "Find Specific Track". Otherwise, they can take advantage of the semantic search portion of our pipline to auto-generate recommended seed tracks by pressing the button labeled "Get Recommended Seeds from Phrase". Pressing this button will open a second window shown below.
<p align="center">
<img src="https://github.com/dymiyata/team-funk-playlist-generator/assets/48339284/78f72656-d4c1-4872-9d34-771a5ae07a2f" width="500" >
</p>
Here, the user is prompted to enter any phrase in order to generated recommended seed tracks from that phrase. Once they enter a phrase, they can press "Generate Recommended Seeds".  Below is an example showing the seeds generated for the phrase "90s rock party".
<p align="center">
<img src="https://github.com/dymiyata/team-funk-playlist-generator/assets/48339284/27430beb-8cb3-4e3c-b6bc-fd2a5a316278" width="500" >
</p>
Once the top 10 recommended seeds are generated, the user can select any number of these recommendations to add to their seed list on the original window.  If the user doesn't like the options presented, they can press "Regenerate" to get the next 10 top recommended seeds from our semantic search model. Below, we show what the original window looks like once some seed tracks are added.  
<p align="center">
<img src="https://github.com/dymiyata/team-funk-playlist-generator/assets/48339284/381e298a-bf18-42bb-855c-75aaa62f3472" width="500" >
</p>
From here the user can generate their full 50 song playlist by pressing the "Generate Playlist from Seeds" button. A new window is then opened with the full list of songs in the newly generated playlist.  The playlist generated from the seed tracks above is shown in the image below. 
<p align="center">
<img src="https://github.com/dymiyata/team-funk-playlist-generator/assets/48339284/1e7cf0d5-fe91-40e3-a60c-c4bbb10b2cb0" width="400" >
</p>
Here only 30 of the 50 tracks are shown, but the user can scroll to see the full playlist. In this window, the user can remove any tracks they do not want and once satisfied, they can save their playlist. 


# Future directions
We plan to deploy this model on an AWS cloud server so that we can collect information from user interaction and feedback with the recommendation engine to further improve our model. Additional layers can also be added to improve recommendations, including adding weights for tracks appearing in playlists that have a high number of followers and incorporating the [MuSe (Musical Sentiment) database](https://www.kaggle.com/datasets/cakiki/muse-the-musical-sentiment-dataset) to further refine recommendations for specific moods or emotions. Finally, we would like to connect our model to the APIs of the most popular streaming platforms in order for users to access their generated playlists wherever they listen to music.
