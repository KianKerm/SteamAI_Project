import requests


# class obj that defines several methods to handle steam urls

class SteamAPI:

    def __init__(self, auth_token):
        self.base_url = 'https://api.steampowered.com/'
        self.auth_token = auth_token

    # =============== STEAM API INTERACTIONS ===============

    def resolve_vanity_url(self, steam_profile_name: str) -> dict:
        '''
        Resolve the vanity url given the profile name to get the user's steamid.
        :param steam_profile_name: like "User01", "xGamerx"
        :return: json containing the steamid of the user
        '''
        api_url = f"{self.base_url}ISteamUser/ResolveVanityURL/v1/?key={self.auth_token}&vanityurl={steam_profile_name}"
        response = requests.get(api_url)
        return response.json()


    def get_owned_games(self, steamid, include_appinfo = True, include_played_free_games = 1) -> dict:
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
                           f"&include_played_free_games={include_played_free_games}")
        response = requests.get(owned_games_url)
        return response.json()

    def get_player_summary(self, steamid):
        '''Supports multiple profiles at same time if passed as a list'''
        player_summary_url = (f"{self.base_url}ISteamUser/GetPlayerSummaries/v0002/?key={self.auth_token}"
                              f"&steamids={steamid}"
                              f"&format=json")
        response = requests.get(player_summary_url)
        return response.json()


    def get_app_list(self,last_appid = '', max_results = 10000) -> dict:
        '''
        last_appid is by default empty string
        :return: all apps available in Steam store.
        '''
        app_list_url = (f"{self.base_url}"
                        f"IStoreService/GetAppList/v1/?key={self.auth_token}"
                        f"&last_appid={last_appid}"
                        f"&max_results={max_results}")
        response = requests.get(app_list_url)
        return response.json()

    def get_app_info(self, appid) -> dict:
        app_info_url = (f"https://store.steampowered.com/api/appdetails?appids={appid}")
        response = requests.get(app_info_url)
        return response.json()