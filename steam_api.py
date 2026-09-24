import requests


# class obj that defines several methods to handle steam urls

class SteamAPI:

    def __init__(self, auth_token):
        self.base_url = 'https://api.steampowered.com/'
        self.auth_token = auth_token

    # for resolving the vanity url or profile name passed by the user

    def resolve_vanity_url(self, userVanityUrl):
        api_url = f"{self.base_url}ISteamUser/ResolveVanityURL/v1/?key={self.auth_token}&vanityurl={userVanityUrl}"
        response = requests.get(api_url)
        return response.json()


    def get_owned_games(self, steamid, include_appinfo = True, include_played_free_games = True):
        '''
        kwargs: steamid The SteamID of the account.
            include_appinfo Include game name and logo information in the output. The default is to return appids only.
            include_played_free_games
            By default, free games like Team Fortress 2 are excluded
            (as technically everyone owns them). If include_played_free_games is set,
            they will be returned if the player has played them at some point.
            This is the same behavior as the games list on the Steam Community.
            steamid
        '''
        owned_games_url = (f"{self.base_url}"
                           f"IPlayerService/GetOwnedGames/v0001/?key={self.auth_token}"
                           f"&steamid={steamid}"
                           f"&format=json"
                           f"&include_appinfo={include_appinfo}"
                           f"&include_played_free_games={include_played_free_games} ")
        response = requests.get(owned_games_url)
        return response.json()

    def get_player_summary(self, steamid):
        '''Supports multiple profiles at same time if pass as comma delimited list'''
        player_summary_url = (f"{self.base_url}ISteamUser/GetPlayerSummaries/v0002/?key={self.auth_token}"
                              f"&steamids={steamid}"
                              f"&format=json")
        response = requests.get(player_summary_url)
        return response.json()

