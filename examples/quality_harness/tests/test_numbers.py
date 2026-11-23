import unittest

from bodhan_eval.numbers import analyze_numbers, compare_numbers, valid_number_finding


class NumbersTest(unittest.TestCase):
    def test_native_attached_suffix_keeps_original_digits_without_assuming_meaning(self):
        text = "Count ১৭টি and ২৫টি."
        tokens = analyze_numbers(text, "translation")["tokens"]
        self.assertEqual([t["value"] for t in tokens], ["17", "25"])
        for token in tokens:
            self.assertEqual(text[slice(*token["span"])], token["text"])
            self.assertEqual(text[slice(*token["suffix_span"])], token["suffix_text"])
            self.assertEqual(token["suffix_interpretation"], "unknown")
        result = compare_numbers("Count 17 and 25.", text)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result.get("candidate") or result.get("candidates"))

    def test_unicode_and_original_spans(self):
        source = "📦 25 kg"
        translation = "📦 ३५ kg"
        result = compare_numbers(source, translation)
        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["verdict"], "finding")
        finding = result["finding"]
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
            ("25 kg", "25000 g"),
        ]:
            with self.subTest(left=left, right=right):
                self.assertEqual(compare_numbers(left, right)["verdict"], "match")

    def test_ambiguous_formats_and_pairings_abstain(self):
        examples = [
            ("1,200 kg", "१.२०० kg"),
            ("0.5 kg", "½ kg"),
            ("25 kg", "25 m"),
            ("Room 25", "Room 35"),
            ("2025-12-04", "2025-12-05"),
            ("3e5 kg", "4e5 kg"),
            ("25 kg", "twenty-five kg"),
            ("3 m/s", "4 m/s"),
            ("ID25", "ID35"),
        ]
        for left, right in examples:
            with self.subTest(left=left, right=right):
                self.assertEqual(compare_numbers(left, right)["status"], "inconclusive")

    def test_multiple_equal_values_cannot_hide_changed_unit_associations(self):
        for source, target in (
            ("A jar holds 7 L and a bag weighs 9 kg.", "7 kg and 9 L"),
            ("The rod is 14 cm and the block weighs 22 g.", "１４ g and २२ cm"),
            ("A 0.4 L bottle and a 3 kg parcel.", "0.4 kg and 3 L"),
        ):
            with self.subTest(source=source):
                result = compare_numbers(source, target)
                self.assertEqual(result["status"], "inconclusive")
                self.assertNotIn("finding", result)
        for source, target in (
            ("7 L and 9 kg", "९ kg and ৭ L"),
            ("14 cm; 22 g", "１４ cm; २२ g"),
            ("0.4 L and 3 kg", "3 kg and ٠٫٤ L"),
        ):
            self.assertEqual(compare_numbers(source, target)["verdict"], "inventory_match")
            self.assertEqual(compare_numbers(source, target)["status"], "inconclusive")

    def test_multi_numeral_and_suffix_limits_do_not_fabricate_quantity_flags(self):
        for source, target in (
            ("A bus has 17 seats. Route 6.", "18 seats. Route 6."),
            ("Page 41 cites section 9.", "Page 42 cites section 9."),
            ("There are 11 birds.", "１１birds"),
            ("The year is 1987.", "1987-th"),
        ):
            self.assertEqual(compare_numbers(source, target)["status"], "inconclusive")
        self.assertEqual(compare_numbers("17 seats. Route 6.", "১৭ seats. Route ৬.")["status"], "inconclusive")
        self.assertEqual(compare_numbers("A page numbered 41.", "A page numbered ४१.")["verdict"], "match")
