import streamlit as st
import pandas as pd
import time
# import numpy as np
from steam_api import *
from data_loader import *
from model import *

st.title('Steam App')

# init steamAPI handler and pass auth token
steam_api_key = st.secrets['steam_api_key']
steam_api_object = SteamAPI(steam_api_key)

# input for steamID
# need to validate text
profile_name = st.text_input(label = 'Enter Steam Name' )
if 'owned_games' not in st.session_state:
    st.session_state.owned_games = None

# chatbot init
if 'AI_helper' not in st.session_state:
    st.session_state.AI_helper = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'steam_df' not in st.session_state:
    st.session_state.steam_df = None

# check for valid string
# nested control flow
if profile_name:

    vanity_response = steam_api_object.resolve_vanity_url(profile_name)['response']
    vanity_success = vanity_response['success']

    if vanity_success == 1:
        st.success('Successfully found your Steam profile!')
        user_steam_id = vanity_response['steamid']
        # set session state so we don't have to get userid each time
        st.session_state.user_steam_id = user_steam_id
        st.write('Get owned games?')

        if st.button('Yes!'):
            owned_games_response = steam_api_object.get_owned_games(user_steam_id)['response']
            # session state for their games owned
            st.session_state.owned_games = owned_games_response
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
            st.write('test owned games',st.session_state.owned_games)
    else:
        st.error('Failed to find your Steam profile!')


if st.session_state.owned_games is not None:
    st.write('Here are your top 10 games by hours played')
    game_df = pd.DataFrame(st.session_state.owned_games['games'])
    # integer div hours
    game_df['playtime_forever'] = game_df['playtime_forever'] // 60
    top10_games = game_df.sort_values('playtime_forever', ascending=False).head(10)
    st.dataframe(top10_games)
    st.bar_chart(data = top10_games, x = 'name' ,y =  'playtime_forever'
                 ,x_label = 'Hours played' ,y_label = 'Game titles'
                 ,horizontal = True ,sort = False ,color = 'name')


# app_id should be a list of app ids
# !!! needs to be rate limited, hard cap of 100k api calls per day,
# therefore it is important that the result is cached properly
# tested caching, 2 seconds to request info for 10 apps down to 0.0009 seconds
@st.cache_data(show_spinner="Fetching data from Steam API...")
def fetch_app_info(app_ids: list, _steam_data):
    '''
    :param app_ids: list of app ids to request from steam API
    :param _steam_data: DataLoader object that will contain data from Steam API
    :return: streamlit dataframe containing all requested app data
    '''

    skipped_app_ids = []
    for app_id in app_ids:
        app_info = steam_api_object.get_app_info(app_id)
        result = (app_info or {}).get(str(app_id)) or {}
        if not result.get('success') or not isinstance(result.get('data'), dict):
            skipped_app_ids.append(app_id)
            continue
        _steam_data.extract_fields_to_pandas(result['data'])

    if skipped_app_ids:
        st.warning(f'Steam returned no store data for {len(skipped_app_ids)} apps; these were skipped.')

    if _steam_data.df is None:
        return pd.DataFrame(columns=['name', 'short_description'])

    # clean text and pass to variable
    # note: some games have less info in detailed_description vs short_description
    # cyberpunk2077 for example has more info stored in images on the steam page
    _steam_data.clean_text()
    streamlit_df = _steam_data.df

    return streamlit_df


owned_games = (st.session_state.owned_games or {}).get('games', [])
if st.button('TryDF', disabled=not owned_games):
    lst = [game['appid'] for game in owned_games]
    steam_data = SteamDataLoader()
    # start = time.time()
    # print(start)
    st.session_state.steam_df = fetch_app_info(lst, steam_data)
    st.session_state.AI_helper = None
    st.session_state.messages = []
    # print(time.time() - start)

if st.session_state.steam_df is not None:
    steam_df = st.session_state.steam_df
    st.dataframe(steam_df)
    if st.session_state.AI_helper is None:
        descriptions = steam_df['short_description']
        valid = descriptions.map(lambda value: isinstance(value, str) and bool(value.strip()))
        if valid.any():
            try:
                with st.spinner('Preparing game descriptions for the chatbot...'):
                    AI_helper = LangModel(
                        dataset=descriptions.loc[valid].tolist(),
                        game_names=steam_df.loc[valid, 'name'].tolist(),
                    )
                    AI_helper.add_chunk_to_database()
                    st.session_state.AI_helper = AI_helper
            except Exception as error:
                st.error(f'Could not prepare the chatbot: {error}')
        else:
            st.warning('No game descriptions are available to search.')

st.subheader('Game chat')
if st.session_state.AI_helper is None:
    st.info('Click TryDF to load game descriptions before chatting.')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

input_query = st.chat_input(
    'Ask me about similar games:',
    disabled=st.session_state.AI_helper is None,
)
if input_query:
    history = list(st.session_state.messages)
    st.session_state.messages.append({'role': 'user', 'content': input_query})
    with st.chat_message('user'):
        st.markdown(input_query)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Chatbot is thinking...'):
                stream = st.session_state.AI_helper.stream_chat(input_query, history=history)
                response = st.write_stream(
                    chunk['message']['content'] for chunk in stream
                )
            st.session_state.messages.append({'role': 'assistant', 'content': response})
        except Exception as error:
            st.error(f'Could not generate a response: {error}')






