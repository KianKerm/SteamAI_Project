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
            st.write(owned_games_response)
    else:
        st.error('Failed to find your Steam profile!')


