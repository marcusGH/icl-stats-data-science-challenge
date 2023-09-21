'''
Contains methods that loads data from data/derived/ (preprocessed data),
adds new informative features to the dataset, and writes back to data/derived/ with
a different name.
'''

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Authenticate with Spotify
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def add_summary_statistics():
    pass


def get_genre(artist, track, album):
    # Search for the track using the given details
    query = f"artist:{artist} track:{track} album:{album}"
    results = sp.search(query)

    # Check if results were found
    if results['tracks']['items']:
        # Get the main artist's ID from the first search result
        artist_id = results['tracks']['items'][0]['artists'][0]['id']

        # Fetch artist details to get the genre
        artist_info = sp.artist(artist_id)

        return artist_info['genres']
    else:
        return None
