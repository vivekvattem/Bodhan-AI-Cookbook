import unittest

from bodhan_eval.numbers import compare_numbers, valid_number_finding


class ResidualPolicyTest(unittest.TestCase):
    def test_independent_clean_controls_never_flag(self):
        controls = [
            ("1000 kg", "1,000 kg"), ("23 kg", "२३ kg"), ("Distance 1 km", "Distance 1000 m"),
            ("A 2 kg B 7 kg", "B 7 kg A 2 kg"), ("There are 8 parcels", "There are eight parcels"),
            ("Date 2026-01-02", "Date 2026-01-02"), ("Area m^2", "Area m^2"),
        ]
        for source, translation in controls:
            with self.subTest(source=source):
                self.assertNotIn("finding", compare_numbers(source, translation))
        self.assertEqual(compare_numbers("Distance 1 km", "Distance 1000 m")["verdict"], "match")

    def test_changed_and_repeated_scaffold_original_spans(self):
        source, translation = "The hive holds 12 jars. [7] [7]", "The hive holds 13 jars. [7] [7]"
        result = compare_numbers(source, translation)
        self.assertEqual(result["verdict"], "finding")
        self.assertTrue(valid_number_finding(result["finding"], {"source": source, "translation": translation}))
        a, b = result["finding"]["span"]
        self.assertEqual(translation[a:b], "13")
        self.assertEqual(compare_numbers("A 2 kg B 5 kg", "A 5 kg B 5 kg")["status"], "inconclusive")
        self.assertEqual(compare_numbers("A 2 kg B 5 kg", "A 3 kg B 7 kg")["status"], "inconclusive")

    def test_unsupported_forms_and_exponent(self):
        for source, translation in [("Value 3e5", "Value 4e5"), ("Area m^2", "Area m^6"),
                                    ("Area m2", "Area m6"), ("1/2", "1/3"), ("1-2", "1-3"),
                                    ("3²", "3³"), ("2**3", "2**4"), ("Room 5", "Room 6")]:
            with self.subTest(source=source):
                result = compare_numbers(source, translation)
                self.assertEqual(result["status"], "inconclusive")
                self.assertNotIn("finding", result)
        self.assertEqual(compare_numbers("Area m2", "Area m6").get("subtype"), "unit_exponent")

    def test_unique_same_script_suffix(self):
        for translation in ("कुल ३५वर्ष", "मात्रा ३५ा", "মোট ৩৫টি", "மொத்தம் ௩௫ஆண்டு", "کل ۳۵سال"):
            with self.subTest(translation=translation):
                result = compare_numbers("Total 25.", translation)
                self.assertEqual(result["verdict"], "finding")
                self.assertEqual(result["finding"]["evidence"]["translation"]["suffix_interpretation"], "unknown")
                a, b = result["finding"]["span"]
                self.assertEqual(len(translation[a:b]), 2)
