import streamlit as st
# import pandas as pd
# import numpy as np
from steam_api import *

st.title('Steam App')

# init steamAPI handler and pass auth token

steam_api_key = st.secrets['steam_api_key']
steam_api_object = SteamAPI(steam_api_key)

# input for steamID
# need to validate text
profile_name = st.text_input(label = 'Enter Steam Name' )

# check for valid string
# nested control flow
if profile_name:

    vanity_response = steam_api_object.resolve_vanity_url(profile_name)['response']
    success = vanity_response['success']

    if success == 1:
        st.success('Successfully found your Steam profile!')
        user_steam_id = vanity_response['steamid']
        st.write('Get owned games?')

        if st.button('Yes!'):
            owned_games_response = steam_api_object.get_owned_games(user_steam_id)['response']
            # if I wanted the avatar you can go thru dict like so ['response']['players'][0]['avatar']
            # probably cleaner way to parse these later on
            ps_r = steam_api_object.get_player_summary(user_steam_id)['response']['players'][0]['avatar']
            st.write('Test psr:', ps_r)
            st.image(ps_r)
            # for games
            # use this url to display image
            # img_icon_url, img_logo_url - these are the filenames of various images for the game.
            # To construct the URL to the image,
            # use this format: https://media.steampowered.com/steamcommunity/public/images/apps/{appid}/{hash}.jpg
            st.write('test owned games',owned_games_response)
    else:
        st.error('Failed to find your Steam profile!')


