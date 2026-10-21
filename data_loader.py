# Load and handle formating for the data pulled in from SteamAPI
import pandas as pd
from html.parser import HTMLParser

# object to clean the html and keep only the text
class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in {'br', 'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in {'p', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            self.parts.append('\n')

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
                          'short_description':data_dict.get('short_description'),
                          'genres':', '.join(genre['description'] for genre in (data_dict.get('genres') or []) if genre.get('description')),
                          'release_date':(data_dict.get('release_date') or {}).get('date'),
                          'steam_appid':data_dict.get('steam_appid'),
                          'developers':next(iter(data_dict.get('developers') or []), None),
                          'publishers':next(iter(data_dict.get('publishers') or []), None)
                          }
        if self.df is None:
            self.df = pd.DataFrame(allowed_fields, index=[0])
        else:
            self.df = pd.concat([self.df ,pd.DataFrame(allowed_fields, index=[self.df.index[-1] + 1]) ] )
        return None

    def clean_text(self):
        """Remove HTML from descriptions in place, preserving missing values."""
        if self.df is None:
            return None

        def remove_html(value):
            if not isinstance(value, str):
                return value
            parser = _HTMLTextExtractor()
            parser.feed(value)
            parser.close()
            return ''.join(parser.parts).strip()

        self.df['detailed_description'] = self.df['detailed_description'].map(remove_html)
        return None

    def add_context(self):
        """Create a labeled FullContext string for each game in place."""
        if self.df is None:
            return None

        columns = ['name', 'detailed_description', 'short_description', 'genres',
                   'release_date', 'developers', 'publishers']
        fields = self.df.reindex(columns=columns).fillna('').astype(str)

        description = (
            fields['detailed_description'].str.strip() + ' ' +
            fields['short_description'].str.strip()
        ).str.strip()
        developer_publisher = (
            fields['developers'].str.strip() + ' / ' + fields['publishers'].str.strip()
        ).str.strip(' /')

        self.df['FullContext'] = (
            'Name: ' + fields['name'] +
            ', Description: ' + description +
            ', Genres: ' + fields['genres'] +
            ', Release date: ' + fields['release_date'] +
            ', Developer and Publisher: ' + developer_publisher + '.'
        )
        if 'playtime_hours' in self.df.columns:
            def format_playtime(hours):
                if pd.isna(hours):
                    return ''
                return f', Playtime hours: {hours:g}'

            self.df['FullContext'] = (
                self.df['FullContext'].str.removesuffix('.') +
                self.df['playtime_hours'].map(format_playtime).astype(str) + '.'
            )
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
