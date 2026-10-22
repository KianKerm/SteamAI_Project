# --- Imports and application setup ---
# Streamlit for the interface; pandas for data management.
# The project classes handle Steam requests, data preparation, and model calls.
import streamlit as st
import pandas as pd
from steam_api import SteamAPI
from data_loader import SteamDataLoader
from models import LangModel
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError

st.title('Steam Personal AI Assistant')

# Read the API key from Streamlit secrets rather than embedding it in source code.
steam_api_object = SteamAPI(st.secrets['steam_api_key'])

# --- Preserve data between Streamlit reruns ---
# Streamlit reruns this script when a widget is submitted. Session state keeps
# the current user's library, OpenAI client, and conversation across reruns.
# Only initialize missing keys so existing values are not overwritten.
for key, default in {
    'owned_games': None,
    'user_steam_id': None,
    'AI_helper': None,
    'messages': [],
    'steam_df': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Load a Steam profile and its owned games ---
# A form submits the profile input together with its button. Steam requests run
# only on submission, rather than each time the user sends a chat message.
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
                # A newly loaded profile invalidates the previous prepared library
                # and its OpenAI context and conversation.
                st.session_state.steam_df = None
                st.session_state.AI_helper = None
                st.session_state.messages = []
                st.success('Steam profile loaded.')
            else:
                st.error('Failed to find your Steam profile!')

# --- Fetch and cache public store information ---
# Cache results for the requested app IDs to avoid repeating store requests.
# The underscore in _steam_data excludes that argument from Streamlit's cache key.
# On a cache hit, the function body is skipped; callers use the returned DataFrame.
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
        # Store responses use string app IDs as keys. Skip unavailable entries
        # so one missing store record does not stop the rest of the library.
        result = (app_info or {}).get(str(app_id)) or {}
        if not result.get('success') or not isinstance(result.get('data'), dict):
            skipped_app_ids.append(app_id)
            continue
        _steam_data.extract_fields_to_pandas(result['data'])

    if skipped_app_ids:
        st.warning(f'Store details unavailable for {len(skipped_app_ids)} apps. Their owned-game records will be retained.')

    # Preserve the columns needed downstream even if every store lookup failed.
    if _steam_data.df is None:
        return pd.DataFrame(columns=['steam_appid', 'name', 'short_description'])

    _steam_data.clean_text()

    return _steam_data.df

# --- Combine store information with this user's playtime ---
# Each item in games is a dictionary; extract its appid for store lookups.
owned_games = (st.session_state.owned_games or {}).get('games', [])

if st.session_state.owned_games is not None:
    st.caption(f'{len(owned_games)} owned games available.')
if st.button('Prepare library', disabled=not owned_games):
    app_ids = [game['appid'] for game in owned_games]
    steam_data = SteamDataLoader()
    # Work on a copy of the returned store data before adding personal fields.
    store_df = fetch_app_info(app_ids, steam_data).copy()
    # Join from the owned-games side so missing store pages do not remove games
    # from counts, rankings, or totals. Personal data stays outside the store cache.
    st.session_state.steam_df = steam_data.combine_library(owned_games, store_df)
    # Recreate the OpenAI helper and clear old messages when the data changes.
    st.session_state.AI_helper = None
    st.session_state.messages = []

# --- Display the user's library and connect it to OpenAI ---
if st.session_state.steam_df is not None:
    steam_df = st.session_state.steam_df
    with st.expander('Your Steam library', expanded=True):
        # Present readable fields; keep internal IDs and model context in steam_df.
        library_columns = {
            'name': 'Game', 'short_description': 'Description', 'genres': 'Genres',
            'release_date': 'Release date', 'developers': 'Developer',
            'publishers': 'Publisher', 'playtime_hours': 'Playtime (hours)',
        }
        st.dataframe(
            steam_df.reindex(columns=library_columns).rename(columns=library_columns),
            hide_index=True,
        )
    # Recreate helpers left in session state by an earlier version of the app.
    if not isinstance(st.session_state.AI_helper, LangModel):
        st.session_state.AI_helper = None
        if 'OPENAI_API_KEY' not in st.secrets:
            st.error('Add OPENAI_API_KEY to Streamlit secrets to enable chat.')
        else:
            try:
                st.session_state.AI_helper = LangModel(
                    library_df=steam_df, api_key=st.secrets['OPENAI_API_KEY'],
                )
            except ValueError as error:
                st.error(str(error))

# --- Display the conversation and handle a new question ---
st.subheader('Game chat')
if st.button('Clear conversation'):
    st.session_state.messages = []
    if st.session_state.AI_helper is not None:
        st.session_state.AI_helper.last_request = None
if st.session_state.AI_helper is None:
    st.info('Load your profile, then click Prepare library before chatting.')

# Redraw saved messages because each interaction reruns the script.
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

input_query = st.chat_input(
    'Ask about your games, playtime, or recommendations:',
    disabled=st.session_state.AI_helper is None,
)
if input_query:
    # Copy earlier messages before saving this question so it is not sent twice.
    history = list(st.session_state.messages)
    st.session_state.messages.append({'role': 'user', 'content': input_query})
    with st.chat_message('user'):
        st.markdown(input_query)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Chatbot is thinking...'):
                # Send the prepared library and recent conversation to OpenAI.
                stream = st.session_state.AI_helper.stream_chat(input_query, history=history)
                # The OpenAI helper yields plain text as it arrives.
                # write_stream returns the complete response for chat history.
                response = st.write_stream(stream)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
        except AuthenticationError:
            st.error('OpenAI authentication failed. Check OPENAI_API_KEY in your secrets.')
        except RateLimitError:
            st.error('OpenAI rate or quota limit reached. Check API billing or retry later.')
        except APIConnectionError:
            st.error('Could not connect to OpenAI. Check your connection and retry.')
        except APIStatusError:
            st.error('OpenAI rejected the request. Check model access and API account settings.')
        except Exception as error:
            st.error(f'Could not generate a response: {error}')

# Uncomment during development to inspect request metadata.
# if st.session_state.AI_helper is not None and st.session_state.AI_helper.last_request is not None:
#     with st.expander('OpenAI request details (debugging)'):
#         st.json(st.session_state.AI_helper.last_request)

