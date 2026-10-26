# --- Imports and application setup ---
# Streamlit for the interface; pandas for data management.
# The project classes handle Steam requests, data preparation, and model calls.
import streamlit as st
import pandas as pd
from pathlib import Path
from requests import RequestException
from steam_api import SteamAPI
from data_loader import SteamDataLoader
from library_insights import show_insights
from models import LangModel
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIStatusError

SESSION_QUESTION_LIMIT = 10

st.set_page_config(page_title='Steam Personal AI Assistant', layout='wide')
with st.sidebar:
    # Resolve the image relative to this file for local and deployed runs.
    st.image(str(Path(__file__).parent / 'images' / 'steamlogo.jpg'), width='stretch')
    st.subheader('Rediscover your library')
    st.write('An OpenAI-powered assistant for insights into your Steam library. '
             'Ask about your playtime, discover overlooked games, or find something '
             'to play based on your mood.')
    st.caption('Load your profile, prepare your library, then start chatting.')
st.title('Steam Personal AI Assistant', text_alignment='center')

# Read API key
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
    'questions_used': 0,
    'chat_error': None,
    'player_summary': None,
    'profile_summary_error': None,
    'profile_background': None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Load a Steam profile and its owned games ---
# A form submits the profile input together with its button. Steam requests run
# only on submission, rather than each time the user sends a chat message.
with st.form('steam_profile'):
    profile_name = st.text_input('Enter SteamID or link your community profile.')
    load_profile = st.form_submit_button('Load profile')

if load_profile:
    if not profile_name.strip():
        st.warning('Enter SteamID or link your community profile.')
    else:
        with st.spinner('Loading Steam profile...'):
            vanity_response = steam_api_object.resolve_vanity_url(profile_name.strip())['response']
            if vanity_response['success'] == 1:
                user_steam_id = vanity_response['steamid']
                owned_games = steam_api_object.get_owned_games(user_steam_id)['response']
                st.session_state.user_steam_id = user_steam_id
                st.session_state.owned_games = owned_games
                # Cosmetic profile data is optional; a failed lookup must not block the library.
                st.session_state.player_summary = None
                st.session_state.profile_summary_error = None
                try:
                    summaries = steam_api_object.get_player_summary(user_steam_id)
                    players = (summaries.get('response') or {}).get('players') or []
                    st.session_state.player_summary = next(
                        (player for player in players if str(player.get('steamid')) == str(user_steam_id)),
                        None,
                    )
                except (RequestException, ValueError):
                    st.session_state.profile_summary_error = 'Profile details could not be loaded. Your library is still available.'
                # Backgrounds are optional and are fetched only when loading a profile.
                st.session_state.profile_background = None
                try:
                    st.session_state.profile_background = steam_api_object.get_profile_background(user_steam_id)
                except (RequestException, ValueError):
                    pass  # Keep the normal app background if Steam cannot provide one.
                # A newly loaded profile invalidates the previous prepared library
                # and its OpenAI context and conversation.
                st.session_state.steam_df = None
                st.session_state.AI_helper = None
                st.session_state.messages = []
                st.success('Steam profile loaded.')
            else:
                st.error('Failed to find your Steam profile!')

# --- Apply the session's profile background ---
background = st.session_state.profile_background
if background:
    # Match the overlay to the active theme so text and widgets stay readable.
    overlay = 'rgba(14, 17, 23, 0.84)' if st.context.theme.type == 'dark' else 'rgba(255, 255, 255, 0.84)'
    st.html(f'''<style>
        .stApp {{
            background-image: linear-gradient({overlay}, {overlay}), url("{background}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    </style>''')
else:
    st.html('<style>.stApp { background-image: none; }</style>')

# --- Display the loaded profile across reruns ---
if st.session_state.user_steam_id is not None:
    player = st.session_state.player_summary or {}
    avatar = player.get('avatarfull') or player.get('avatarmedium') or player.get('avatar')
    with st.container(border=True):
        avatar_column, name_column = st.columns([1, 8], vertical_alignment='center')
        if avatar:
            avatar_column.image(avatar, width=80)
        name_column.text(player.get('personaname') or f"Steam profile {st.session_state.user_steam_id}")
    if st.session_state.profile_summary_error:
        st.caption(st.session_state.profile_summary_error)

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
    library_tab, insights_tab = st.tabs(['Your library', 'Insights'])
    with library_tab:
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
    with insights_tab:
        show_insights(steam_df)
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
    st.session_state.chat_error = None
    if st.session_state.AI_helper is not None:
        st.session_state.AI_helper.last_request = None
if st.session_state.AI_helper is None:
    st.info('Load your profile, then click Prepare library before chatting.')

# Session allowance counts submitted requests, including failed attempts.
# Clearing history or loading another profile does not replenish it.
questions_remaining = max(0, SESSION_QUESTION_LIMIT - st.session_state.questions_used)
st.caption(f'{questions_remaining} of {SESSION_QUESTION_LIMIT} questions remaining this session.')
if questions_remaining == 0:
    st.info('You have reached the 10-question limit for this session. You can still explore your library and insights.')
if st.session_state.chat_error:
    st.error(st.session_state.chat_error)

# Redraw saved messages because each interaction reruns the script.
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

input_query = st.chat_input(
    'Ask about your games, playtime, or recommendations:',
    disabled=st.session_state.AI_helper is None or questions_remaining == 0,
)
if input_query and input_query.strip() and st.session_state.AI_helper is not None and questions_remaining > 0:
    # Enforce the limit before any model call, independently of the disabled widget.
    st.session_state.questions_used += 1
    st.session_state.chat_error = None
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
            st.session_state.chat_error = 'OpenAI authentication failed. Check OPENAI_API_KEY in your secrets.'
        except RateLimitError:
            st.session_state.chat_error = 'OpenAI rate or quota limit reached. Check API billing or retry later.'
        except APIConnectionError:
            st.session_state.chat_error = 'Could not connect to OpenAI. Check your connection and retry.'
        except APIStatusError:
            st.session_state.chat_error = 'OpenAI rejected the request. Check model access and API account settings.'
        except Exception as error:
            st.session_state.chat_error = f'Could not generate a response: {error}'
    # Refresh the allowance and disabled input after either success or failure.
    st.rerun()

# Uncomment during development to inspect request metadata.
# if st.session_state.AI_helper is not None and st.session_state.AI_helper.last_request is not None:
#     with st.expander('OpenAI request details (debugging)'):
#         st.json(st.session_state.AI_helper.last_request)

with st.sidebar:
    st.divider()
    st.caption('Disclaimer: this application is not endorsed or a product of steam or valve corporation, it is just a fan-made project for fun.')

