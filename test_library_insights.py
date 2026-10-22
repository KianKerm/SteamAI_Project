"""Check genre overlap and exclusive playtime groups with synthetic records."""
import unittest

import pandas as pd

from library_insights import build_insights


class LibraryInsightsTests(unittest.TestCase):
    def test_genres_receive_full_hours_without_duplicate_membership(self):
        library = pd.DataFrame({
            'steam_appid': [1, 2, 3, 4, 5, 6],
            'genres': ['Action, Adventure, Action', 'Action', 'Adventure', None, 'Puzzle', 'Puzzle'],
            'playtime_hours': [10, 0, 1.5, None, 2, 0],
        })
        result = build_insights(library)
        self.assertEqual(result['hours'], 13.5)
        self.assertEqual(result['unplayed'], 2)
        self.assertEqual(result['brief'], 1)
        self.assertEqual(result['unknown'], 1)
        genre_hours = result['genre_hours'].set_index('Genre')['Hours']
        self.assertEqual(genre_hours['Action'], 10)
        self.assertEqual(genre_hours['Adventure'], 11.5)
        self.assertEqual(result['unplayed_genres'].set_index('Genre').loc['Puzzle', 'Games'], 1)
        self.assertEqual(result['breakdown']['Games'].tolist(), [2, 1, 1, 1, 1])
        self.assertEqual(result['breakdown']['Games'].sum(), len(library))

    def test_missing_playtime_is_not_zero_and_empty_is_supported(self):
        library = pd.DataFrame({'steam_appid': [1], 'genres': [None], 'playtime_hours': [None]})
        result = build_insights(library)
        self.assertIsNone(result['hours'])
        self.assertEqual(result['unplayed'], 0)
        self.assertTrue(result['genre_hours'].empty)
        self.assertEqual(build_insights(library.iloc[:0])['games'], 0)


if __name__ == '__main__':
    unittest.main()
