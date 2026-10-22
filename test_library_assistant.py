"""Offline regression checks: python -m unittest test_library_assistant."""
import json
import unittest
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from data_loader import SteamDataLoader
from model import LangModel, QueryPlan


class LibraryAssistantTests(unittest.TestCase):
    def setUp(self):
        self.owned = [
            {'appid': 1, 'name': 'Alpha', 'playtime_forever': 90},
            {'appid': 2, 'name': 'Beta', 'playtime_forever': 0},
            {'appid': 3, 'name': 'Missing store', 'playtime_forever': 300},
            {'appid': 4, 'name': 'Unknown hours'},
        ]
        self.store = pd.DataFrame([
            {'steam_appid': 2, 'name': 'Beta', 'short_description': '<p>Explore</p>',
             'genres': 'Adventure', 'release_date': 'Jan 1, 2020', 'developers': 'Dev B'},
            {'steam_appid': 1, 'name': 'Alpha', 'short_description': 'Explore space',
             'genres': 'Adventure', 'release_date': 'Jan 1, 2010', 'developers': 'Dev A'},
        ])
        self.frame = SteamDataLoader().combine_library(self.owned, self.store)
        self.model = LangModel(self.frame)

    def run_plan(self, **kwargs):
        return self.model.execute_plan(QueryPlan(**kwargs))

    def test_merge_preserves_missing_store_games_and_playtime(self):
        rows = self.frame.set_index('steam_appid')
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows.loc[1, 'playtime_hours'], 1.5)
        self.assertEqual(rows.loc[3, 'name'], 'Missing store')
        self.assertFalse(rows.loc[3, 'store_data_available'])
        self.assertTrue(pd.isna(rows.loc[4, 'playtime_hours']))
        self.assertEqual(rows.loc[2, 'short_description'], 'Explore')
        self.assertNotIn('Playtime hours:', rows.loc[4, 'FullContext'])
        self.assertNotIn('playtime_hours', self.store)

    def test_all_store_requests_missing_and_empty_library(self):
        empty_store = pd.DataFrame(columns=['steam_appid', 'name', 'short_description'])
        frame = SteamDataLoader().combine_library(self.owned, empty_store)
        self.assertEqual(len(frame), 4)
        self.assertEqual(LangModel(frame).execute_plan(QueryPlan(operation='count'))['value'], 4)
        empty = SteamDataLoader().combine_library([], empty_store)
        self.assertEqual(LangModel(empty).execute_plan(QueryPlan(operation='count'))['value'], 0)

    def test_counts_and_aggregates_include_missing_store_records(self):
        self.assertEqual(self.run_plan(operation='count')['value'], 4)
        total = self.run_plan(operation='sum')
        self.assertEqual(total['value'], 6.5)
        self.assertEqual(total['unknown_playtime_count'], 1)
        self.assertAlmostEqual(self.run_plan(operation='average')['value'], 6.5 / 3)

    def test_zero_is_not_missing(self):
        result = self.run_plan(operation='count', filters=[
            {'field': 'playtime_hours', 'operator': 'eq', 'value': 0}])
        self.assertEqual(result['value'], 1)

    def test_ranking_and_follow_up(self):
        result = self.run_plan(operation='list', sort_by='playtime_hours', descending=True, limit=2)
        self.assertEqual([r['steam_appid'] for r in result['records']], [3, 1])
        self.assertEqual(result['unknown_sort_values'], 1)
        self.assertTrue(result['truncated'])
        follow_up = self.run_plan(operation='list', scope='previous', sort_by='release_date', limit=1)
        self.assertEqual(follow_up['records'][0]['name'], 'Alpha')

    def test_exact_title_case_insensitive_and_missing_title_clarifies(self):
        result = self.run_plan(operation='lookup', titles=[' alpha '])
        self.assertEqual(result['records'][0]['developers'], 'Dev A')
        with self.assertRaises(ValueError):
            self.run_plan(operation='lookup', titles=['Unknown title'])

    def test_filters_and_chronological_order(self):
        result = self.run_plan(operation='list', sort_by='release_date', filters=[
            {'field': 'genres', 'operator': 'contains', 'value': 'adventure'}])
        self.assertEqual([r['name'] for r in result['records']], ['Alpha', 'Beta'])
        result = self.run_plan(operation='count', filters=[
            {'field': 'release_date', 'operator': 'lt', 'value': '2015-01-01'}])
        self.assertEqual(result['value'], 1)

    def test_semantic_filtering_and_app_id_mapping(self):
        with patch('model.ollama.embed', return_value={'embeddings': [[1.0, 0.0]]}) as embed:
            result = self.run_plan(operation='semantic', search_query='exploration', filters=[
                {'field': 'playtime_hours', 'operator': 'eq', 'value': 0}])
        self.assertEqual([r['steam_appid'] for r in result['records']], [2])
        self.assertEqual(set(self.model.VECTOR_DB), {1, 2})
        self.assertNotIn('Playtime hours', embed.call_args_list[0].kwargs['input'])

    def test_similar_games_exclude_source(self):
        with patch('model.ollama.embed', return_value={'embeddings': [[1.0, 0.0]]}):
            result = self.run_plan(operation='semantic', source_title='Alpha')
        self.assertEqual([r['steam_appid'] for r in result['records']], [2])

    def test_invalid_operations_and_filters_are_rejected(self):
        with self.assertRaises(ValidationError):
            QueryPlan(operation='execute_python')
        with self.assertRaises(ValidationError):
            QueryPlan(operation='list', limit=1000)
        with self.assertRaises(ValueError):
            self.run_plan(operation='count', filters=[
                {'field': 'name', 'operator': 'gt', 'value': 'Alpha'}])

    def test_stream_uses_plan_history_and_computed_evidence_without_embeddings(self):
        plan = QueryPlan(operation='sum')
        history = [{'role': 'user', 'content': 'My library'}]
        with patch('model.ollama.chat', side_effect=[
            {'message': {'content': plan.model_dump_json()}},
            iter([{'message': {'content': '6.5 hours'}}]),
        ]) as chat, patch('model.ollama.embed') as embed:
            output = list(self.model.stream_chat('Total hours?', history))
        self.assertEqual(output[0]['message']['content'], '6.5 hours')
        self.assertEqual(self.model.last_result['value'], 6.5)
        self.assertIn('"value": 6.5', chat.call_args.kwargs['messages'][0]['content'])
        self.assertEqual(chat.call_args_list[0].kwargs['messages'][1], history[0])
        embed.assert_not_called()

    def test_malformed_plan_and_unsupported_data_do_not_invent_results(self):
        with patch('model.ollama.chat', side_effect=[
            {'message': {'content': 'not JSON'}}, iter([]),
        ]):
            list(self.model.stream_chat('Question'))
        self.assertIn('clarification', self.model.last_result)
        result = self.run_plan(operation='unsupported', clarification='Achievements are unavailable.')
        self.assertEqual(result['clarification'], 'Achievements are unavailable.')


if __name__ == '__main__':
    unittest.main()
