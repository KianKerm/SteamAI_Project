"""Offline tests: no real secrets or OpenAI requests are used."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from data_loader import SteamDataLoader
from models import LangModel


def sample_library():
    owned = [
        {'appid': 1, 'name': 'Oxygen Not Included', 'playtime_forever': 600},
        {'appid': 2, 'name': 'Lethal Company', 'playtime_forever': 300},
        {'appid': 3, 'name': 'Catan Universe', 'playtime_forever': 60},
        {'appid': 4, 'name': 'Unknown playtime'},
        {'appid': 5, 'name': 'Unplayed', 'playtime_forever': 0},
    ]
    store = pd.DataFrame([{'steam_appid': 1, 'name': 'Oxygen Not Included',
                          'short_description': 'Manage a colony', 'genres': 'Simulation'}])
    return SteamDataLoader().combine_library(owned, store)


def fake_stream(text='A grounded answer', status='completed'):
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter([
        SimpleNamespace(type='response.created'),
        SimpleNamespace(type='response.output_text.delta', delta=text),
    ])
    stream.get_final_response.return_value = SimpleNamespace(status=status, usage=None)
    return stream


class OpenAIModelTests(unittest.TestCase):
    def setUp(self):
        self.factory = patch('models.OpenAI')
        self.client = self.factory.start().return_value
        self.addCleanup(self.factory.stop)
        self.model = LangModel(sample_library(), api_key='fake-test-key')

    def test_full_library_and_summary_preserve_missing_values(self):
        context = self.model.library_context
        self.assertEqual(context['summary']['owned_game_count'], 5)
        self.assertEqual(context['summary']['total_known_playtime_hours'], 16)
        self.assertEqual(context['summary']['average_known_playtime_hours'], 4)
        self.assertEqual(context['summary']['unknown_playtime_count'], 1)
        self.assertEqual(context['summary']['zero_playtime_count'], 1)
        self.assertEqual([r['name'] for r in context['games'][:3]],
                         ['Oxygen Not Included', 'Lethal Company', 'Catan Universe'])
        self.assertEqual([r['playtime_rank'] for r in context['games'][:3]], [1, 2, 3])
        self.assertIsNone(context['games'][-1]['playtime_hours'])
        self.assertFalse(context['games'][1]['store_data_available'])

    def test_stream_payload_contains_all_records_and_follow_up_history(self):
        self.client.responses.stream.return_value = fake_stream('Library answer')
        history = [{'role': 'user', 'content': 'Show my games'},
                   {'role': 'assistant', 'content': 'Earlier answer'},
                   {'role': 'system', 'content': 'Do not forward this'}]
        self.assertEqual(list(self.model.stream_chat('Which of those?', history)), ['Library answer'])
        request = self.client.responses.stream.call_args.kwargs
        self.assertFalse(request['store'])
        self.assertEqual(request['model'], 'gpt-4.1-mini')
        self.assertIn('Catan Universe', request['input'][0]['content'])
        self.assertEqual(len(request['input']), 4)
        self.assertEqual(self.model.last_request['status'], 'completed')
        self.assertNotIn('fake-test-key', json.dumps(self.model.last_request))

    def test_incomplete_response_is_not_silently_successful(self):
        self.client.responses.stream.return_value = fake_stream(status='incomplete')
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            list(self.model.stream_chat('Question'))
        self.assertEqual(self.model.last_request['status'], 'failed')

    def test_oversized_input_is_rejected_before_api_call(self):
        self.model.MAX_INPUT_CHARACTERS = 10
        with self.assertRaisesRegex(ValueError, 'too large'):
            list(self.model.stream_chat('Question'))
        self.client.responses.stream.assert_not_called()

    def test_empty_library_and_duplicate_ids(self):
        empty = self.model.library.iloc[:0]
        model = LangModel(empty, api_key='fake-test-key')
        self.assertEqual(model.library_context['games'], [])
        self.assertIsNone(model.library_context['summary']['total_known_playtime_hours'])
        with self.assertRaisesRegex(ValueError, 'unique'):
            LangModel(pd.concat([self.model.library, self.model.library]), api_key='fake-test-key')

    def test_streamlit_profile_prepare_chat_and_clear(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from steam_api import SteamAPI
        # Replace the entire secrets object BEFORE running the app, so the real
        # secrets.toml is never loaded by application code or the test harness.
        with patch.object(st, 'secrets', {'steam_api_key': 'fake-steam', 'OPENAI_API_KEY': 'fake-test-key'}), \
             patch.object(SteamAPI, 'resolve_vanity_url', return_value={'response': {'success': 1, 'steamid': '123'}}), \
             patch.object(SteamAPI, 'get_owned_games', return_value={'response': {'games': [
                 {'appid': 1, 'name': 'Example', 'playtime_forever': 90}]}}), \
             patch.object(SteamAPI, 'get_app_info', return_value={'1': {'success': False}}):
            self.client.responses.stream.return_value = fake_stream('Example: 1.5 hours')
            app = AppTest.from_file('main.py').run()
            self.assertFalse(app.exception)
            app.text_input[0].set_value('example')
            app.button[0].click().run()
            app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state['steam_df']), 1)
            self.assertEqual(app.dataframe[0].value.columns.tolist(),
                             ['Game', 'Description', 'Genres', 'Release date',
                              'Developer', 'Publisher', 'Playtime (hours)'])
            app.chat_input[0].set_value('What is my playtime?').run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['messages'][-1]['content'], 'Example: 1.5 hours')
            self.assertEqual(len(app.get('json')), 0)
            app.button[2].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['messages'], [])


if __name__ == '__main__':
    unittest.main()
