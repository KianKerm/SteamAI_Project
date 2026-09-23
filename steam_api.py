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


    def get_owned_games(self, steamid):
        owned_games_url = f"{self.base_url}/IPlayerService/GetOwnedGames/v0001/?key={self.auth_token}&steamid={steamid}&format=json"
        response = requests.get(owned_games_url)
        # response.status_code

        return response.json()