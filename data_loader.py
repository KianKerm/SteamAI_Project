# Load and handle formating for the data pulled in from SteamAPI
import pandas as pd

class SteamDataLoader:

    def __init__(self):
        self.df = None


    def extract_fields_to_pandas(self, data: dict, appid) -> pd.DataFrame:
        x = None
        # detailed_description (req regex)
        # short_description
        # genres (exclude free to play) [genres].keys()
        # dataframe should look like
        # Name | short_description | genre 1 | 2




# step 1: iterate through all owned games ad get appid, playtime forever in hours
# step 2: merge with appid, the short_description and genres from another api call
# tag each row with unique id
# loop over all owned games (use st.cache_data)
# preprocess the short_description (all UPPER no punctuation)
# agg all genres into a list so that df is N x 5
# should use relative library playtime to weight reccommendations (not sure how yet)
# model shhould one hot encode all genres

# stochastically pick games from steam marketplace and ocmpute similarity rathe than procesingg 10000+ api calls in one inute
