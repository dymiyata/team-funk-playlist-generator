# Playlist Recommendation Engine

Digital audio streaming is on the rise, with a reported 61% of U.S. listeners streaming audio in 2021 ([MRC data](https://static.billboard.com/files/2021/09/U.S.-Music-360-2021-Draft-Report_Sneak-Preview_Sept-2021-2-1631178109.pdf)), and audio song streaming increasing globally by 25.6% between 2021 and 2022 ([Luminate 2022](https://luminatedata.com/reports/luminate-2022-u-s-year-end-report/?aliId=eyJpIjoieEN0ZFVqeklFU0RkYTZNeSIsInQiOiJhMFpUMXVSSHdEWlFNS1Rkcms2cDFnPT0ifQ%253D%253D)). Playlists play an increasingly important role in curating and organizing digital listening experiences, with a 2017 survey by [Nielsen Music](https://www.nielsen.com/insights/2017/music-360-2017-highlights/) finding that 75% of listeners who stream music online use playlists and over half create their own. As playlist creation and curation is often a time consuming process, users frequently turn to track recommendation systems to help them in constructing new playlists ([Inman et al., 2020](https://www.proquest.com/openview/fc6da445396fb9af102c5e8b4226db0d/1?pq-origsite=gscholar&cbl=30100)). Most existing track recommendation systems rely on listeners to provide seed tracks, and then utilize a variety of different approaches to recommend additional tracks in either a playlist-like listening session or as sequential track recommendations based on user feedback.

Our recommendation engine takes a different approach, allowing listeners to generate playlists based on a semantic string, such as the title of desired playlist, specific mood (happy, relaxed), atmosphere (tropical), or function (party music, focus) [this needs work]. Using a publicly available dataset of [existing playlists](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge), we combine a semantic similarity vector model with a matrix factorization model to allow users to quickly and easily generate playlists to fit any occasion.



# Data Processing

The [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) (or MPD for short) is a publicly available compilation of 1 million playlists created by Spotify users. It consists of 1,000 json files, each containing 1,000 playlists. A playlist consists of metadata (e.g. a unique playlist id, name, number of followers, number of tracks, etc.) and a list of tracks with their respetive metadata (e.g. name, artist, album, etc.). Given that a track usually appears in multiple playlists, we created a SQL database to avoid storing redundant metadata. This is our schema:

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

**Measure of success:** Storing the MPD requires 5.8 GB when compressed in a .zip folder and 33.5 GB when uncompressed. Our SQL database uses only 1.6 GB.



# Semantic Search



# Matrix Factorization Training



# GUI



# Future directions

