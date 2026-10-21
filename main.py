import streamlit as st
import pandas as pd
from steam_api import SteamAPI
from data_loader import SteamDataLoader
from model import LangModel

st.title('Steam App')

steam_api_object = SteamAPI(st.secrets['steam_api_key'])

for key, default in {
    'owned_games': None,
    'user_steam_id': None,
    'AI_helper': None,
    'messages': [],
    'steam_df': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.form('steam_profile'):
    profile_name = st.text_input('Enter Steam profile name')
    load_profile = st.form_submit_button('Load profile')

if load_profile:
    if not profile_name.strip():
        st.warning('Enter a Steam profile name.')
    else:
        with st.spinner('Loading Steam profile...'):
            vanity_response = steam_api_object.resolve_vanity_url(profile_name.strip())['response']
            if vanity_response['success'] == 1:
                user_steam_id = vanity_response['steamid']
                owned_games = steam_api_object.get_owned_games(user_steam_id)['response']
                st.session_state.user_steam_id = user_steam_id
                st.session_state.owned_games = owned_games
                st.session_state.steam_df = None
                st.session_state.AI_helper = None
                st.session_state.messages = []
                st.success('Steam profile loaded.')
            else:
                st.error('Failed to find your Steam profile!')

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
        return pd.DataFrame(columns=['steam_appid', 'name', 'short_description'])

    _steam_data.clean_text()

    return _steam_data.df

# owned games
owned_games = (st.session_state.owned_games or {}).get('games', [])

if st.session_state.owned_games is not None:
    st.caption(f'{len(owned_games)} owned games available.')
if st.button('Prepare library', disabled=not owned_games):
    app_ids = [game['appid'] for game in owned_games]
    steam_data = SteamDataLoader()
    store_df = fetch_app_info(app_ids, steam_data).copy()
    # Keep user-specific playtime outside the cached store-data request.
    playtime_df = pd.DataFrame(owned_games).reindex(columns=['appid', 'playtime_forever'])
    playtime_df = playtime_df.rename(columns={'appid': 'steam_appid'})
    store_df['steam_appid'] = pd.to_numeric(store_df['steam_appid'], errors='raise').astype('Int64')
    playtime_df['steam_appid'] = pd.to_numeric(playtime_df['steam_appid'], errors='raise').astype('Int64')
    # Steam reports minutes. Retain fractional hours and distinguish missing from zero.
    playtime_df['playtime_hours'] = pd.to_numeric(
        playtime_df['playtime_forever'], errors='coerce'
    ) / 60
    steam_data.df = store_df.merge(
        playtime_df[['steam_appid', 'playtime_hours']],
        on='steam_appid', how='left', validate='one_to_one',
    )
    steam_data.add_context()
    st.session_state.steam_df = steam_data.df
    st.session_state.AI_helper = None
    st.session_state.messages = []

if st.session_state.steam_df is not None:
    steam_df = st.session_state.steam_df
    with st.expander('Library DataFrame (debugging)', expanded=True):
        st.dataframe(steam_df)
    if st.session_state.AI_helper is None:
        descriptions = steam_df['FullContext']
        valid = descriptions.map(lambda value: isinstance(value, str) and bool(value.strip()))
        if valid.any():
            try:
                with st.spinner('Preparing game descriptions for the chatbot...'):
                    AI_helper = LangModel(
                        dataset=descriptions.loc[valid].tolist(),
                    )
                    AI_helper.add_chunk_to_database()
                    st.session_state.AI_helper = AI_helper
            except Exception as error:
                st.error(f'Could not prepare the chatbot: {error}')
        else:
            st.warning('No game descriptions are available to search.')

st.subheader('Game chat')
if st.session_state.AI_helper is None:
    st.info('Load your profile, then click Prepare library before chatting.')

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

