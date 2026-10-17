# Load and handle formating for the data pulled in from SteamAPI
import pandas as pd

class SteamDataLoader:

    def __init__(self):
        self.df = None


    def extract_fields_to_pandas(self, data_dict: dict):
        # detailed_description (req regex)
        # genres (exclude free to play) [genres].keys()
        # dataframe should look like
        # Name | short_description | genre 1 | 2
        # dict is from here steam_app_info['3513350']['data']

        allowed_fields = {'name':data_dict.get('name'),
                          'detailed_description':data_dict.get('detailed_description'),
                          'genres':', '.join([data_dict.get('genres')[idx]['description'] for idx in range(len(data_dict.get('genres')))]),
                          'release_date':data_dict.get('release_date')['date'],
                          'steam_appid':data_dict.get('steam_appid'),
                          'developers':data_dict.get('developers')[0],
                          'publishers':data_dict.get('publishers')[0]
                          }
        if self.df is None:
            self.df = pd.DataFrame(allowed_fields, index=[0])
        else:
            self.df = pd.concat([self.df ,pd.DataFrame(allowed_fields, index=[self.df.index[-1] + 1]) ] )
        return None




# step 1: iterate through all owned games ad get appid, playtime forever in hours
# step 2: merge with appid, the short_description and genres from another api call
# tag each row with unique id
# loop over all owned games (use st.cache_data)
# preprocess the short_description (all UPPER no punctuation)
# agg all genres into a list so that df is N x 5
# should use relative library playtime to weight reccommendations (not sure how yet)
# model shhould one hot encode all genres

# stochastically pick games from steam marketplace and ocmpute similarity rathe than procesingg 10000+ api calls in one inute
