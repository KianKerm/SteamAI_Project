import requests
from urllib.parse import urlparse


# class obj that defines several methods to handle steam urls

class SteamAPI:

    def __init__(self, auth_token):
        self.base_url = 'https://api.steampowered.com/'
        self.auth_token = auth_token

    # =============== STEAM API INTERACTIONS ===============

    def resolve_vanity_url(self, steam_profile_name: str) -> dict:
        '''
        Resolve the vanity url given the profile name to get the user's steamid.
        :param steam_profile_name: a vanity name, SteamID64, or Steam community profile URL
        :return: json containing the steamid of the user
        '''
        steam_profile_name = steam_profile_name.strip()
        path_prefix = None
        if '://' in steam_profile_name:
            profile = urlparse(steam_profile_name)
            # Ignore query strings and take the final non-trailing path segment.
            path_prefix, _, steam_profile_name = profile.path.rstrip('/').rpartition('/')
            if (profile.scheme not in {'https', 'http'} or
                    profile.hostname != 'steamcommunity.com' or
                    path_prefix not in {'/id', '/profiles'} or not steam_profile_name):
                return {'response': {'success': 42, 'message': 'Invalid Steam profile URL.'}}

        is_steam_id = (steam_profile_name.isascii() and steam_profile_name.isdigit()
                       and len(steam_profile_name) == 17)
        if path_prefix == '/profiles' and not is_steam_id:
            return {'response': {'success': 42, 'message': 'Invalid SteamID64.'}}
        # Keep identifiers as strings. Even numeric /id/ names require vanity lookup.
        if is_steam_id and path_prefix != '/id':
            return {'response': {'success': 1, 'steamid': steam_profile_name}}
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
        response = requests.get(player_summary_url, timeout=15)
        response.raise_for_status()
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
