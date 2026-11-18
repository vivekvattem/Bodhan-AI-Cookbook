import unittest

from bodhan_eval.numbers import compare_numbers, valid_number_finding


class NumbersTest(unittest.TestCase):
    def test_unicode_and_original_spans(self):
        source = "📦 25 kg"
        translation = "📦 ३५ kg"
        result = compare_numbers(source, translation)
        finding = result["candidate"]
        self.assertEqual(finding["span"], [2, 4])
        self.assertEqual(finding["evidence"]["source"]["text"], "25")
        self.assertEqual(finding["evidence"]["translation"]["text"], "३५")
        self.assertTrue(
            valid_number_finding(
                finding, {"source": source, "translation": translation}
            )
        )
        self.assertFalse(
            valid_number_finding(finding, {"source": source, "translation": "📦 २५ kg"})
        )

    def test_equal_decimal_values(self):
        for left, right in [
            ("25 kg", "२५ kg"),
            ("2.50 L", "௨.௫ L"),
            ("-3.2 m", "−٣٫٢ m"),
        ]:
            with self.subTest(left=left, right=right):
                self.assertEqual(compare_numbers(left, right)["verdict"], "match")

    def test_ambiguous_formats_and_pairings_abstain(self):
        examples = [
            ("1,200 kg", "१.२०० kg"),
            ("0.5 kg", "½ kg"),
            ("25 kg", "25 m"),
            ("25 kg", "25000 g"),
            ("Room 25", "Room 35"),
            ("2025-12-04", "2025-12-05"),
            ("3e5 kg", "4e5 kg"),
            ("25 kg", "twenty-five kg"),
            ("3 m/s", "4 m/s"),
            ("2 kg and 3 kg", "2 kg and 4 kg"),
            ("ID25", "ID35"),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                self.assertEqual(compare_numbers(left, right)["status"], "inconclusive")
