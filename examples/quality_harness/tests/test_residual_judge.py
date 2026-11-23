import unittest

from bodhan_eval.residual_judge import build_request, validate_review


class ResidualJudgeTest(unittest.TestCase):
    def test_request_builder_checks_original_spans_and_allowlists_metadata(self):
        source, translation = 'Use 25 units.', '३५ units.'
        result = build_request(source, translation,
                               [{'span': [4, 6], 'value': '25', 'labels': 'do-not-send'}],
                               [{'span': [0, 2], 'value': '35', 'injection': 'do-not-send'}])
        self.assertEqual(result['source_tokens'][0]['id'], 'S1')
        self.assertEqual(result['translation_tokens'][0]['id'], 'T1')
        self.assertTrue(result['translation_tokens'][0]['context'].isascii())
        self.assertNotIn('do-not-send', str(result))
        with self.assertRaises(ValueError):
            build_request(source, translation, [{'span': [4, 7], 'value': '25'}], [])

    def setUp(self):
        self.source = [{'id': 'S1', 'value': '3'}, {'id': 'S2', 'value': '5'}]
        self.target = [{'id': 'T1', 'value': '4'}, {'id': 'T2', 'value': '5'}]
        self.answer = {'pairs': [
            {'source': 'S1', 'translation': 'T1', 'verdict': 'mismatch', 'reason': 'The value differs.'},
            {'source': 'S2', 'translation': 'T2', 'verdict': 'uncertain', 'reason': 'The pairing is unclear.'},
        ]}

    def test_valid_and_uncertain(self):
        self.assertEqual(validate_review(self.answer, self.source, self.target), self.answer['pairs'])

    def test_reject_unknown_duplicate_and_missing_ids(self):
        for rows in (
            [{**self.answer['pairs'][0], 'source': 'S9'}, self.answer['pairs'][1]],
            [self.answer['pairs'][0], {**self.answer['pairs'][1], 'source': 'S1'}],
            self.answer['pairs'][:1],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_review({'pairs': rows}, self.source, self.target)

    def test_equal_value_mismatch_and_indic_output_rejected(self):
        for row in ({**self.answer['pairs'][1], 'verdict': 'mismatch'},
                    {**self.answer['pairs'][1], 'reason': 'पांच'}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_review({'pairs': [self.answer['pairs'][0], row]}, self.source, self.target)

    def test_explicit_unpaired_covers_each_side(self):
        rows = [{'source': 'S1', 'translation': None, 'verdict': 'unpaired', 'reason': 'No pair.'},
                {'source': 'S2', 'translation': 'T2', 'verdict': 'match', 'reason': 'Same value.'},
                {'source': None, 'translation': 'T1', 'verdict': 'unpaired', 'reason': 'No pair.'}]
        self.assertEqual(validate_review({'pairs': rows}, self.source, self.target), rows)
