import unittest

from bodhan_eval.numbers import compare_numbers, quantity_candidates
from bodhan_eval.semantic import finalize_judgment, parse_judgment


def judgment(source_span, target_span, **changes):
    source = {"field": "source", "span": source_span}
    target = {"field": "translation", "span": target_span}
    finding = {"category": "number", "severity": "major", **target,
               "explanation": "The corresponding quantity changed.",
               "evidence": [source, target], "confidence": "high", "uncertainty": ""}
    finding.update(changes)
    return parse_judgment({"verdict": "findings", "explanation": "Compared the quantities.", "findings": [finding]})


class ContextNumberTest(unittest.TestCase):
    def test_equal_inventory_does_not_prove_association(self):
        for target in ("A 5 kg, B 2 kg", "B 5 kg, A 2 kg"):
            check = compare_numbers("A 2 kg, B 5 kg", target)
            self.assertEqual(check["status"], "inconclusive")
            self.assertNotIn("finding", check)

    def test_incompatible_or_opaque_units_are_not_confirmed(self):
        for target_text in ("३५টি", "३५ seconds"):
            item = {"source": "Use 25 kg.", "translation": target_text}
            source = {"field": "source", "quantity_id": "S0"}
            target = {"field": "translation", "quantity_id": "T0"}
            finding = {"category": "number", "severity": "major", **target,
                       "explanation": "The quantity changed.", "confidence": "high", "uncertainty": "",
                       "evidence": [source, target, {"field": "source", "quote": "25 kg"}]}
            raw = {"verdict": "findings", "explanation": "Compared.", "findings": [finding]}
            result, flags = finalize_judgment(parse_judgment(raw), item)
            self.assertEqual((result["status"], flags), ("inconclusive", []))

    def test_candidates_preserve_original_native_digits_and_skip_fractions(self):
        text = "Use २५টি, 3/4 and 12kg."
        candidates = quantity_candidates(text, "translation")
        self.assertEqual([text[slice(*item["span"])] for item in candidates], ["२५", "12"])
        self.assertEqual([item["id"] for item in candidates], ["T0", "T1"])

    def test_number_ids_resolve_exact_spans_and_reject_bad_or_equal_pairs(self):
        item = {"source": "Use 12 g salt and 25 ml water.", "translation": "पानी ३५ ml और नमक १२ g"}
        source = {"field": "source", "quantity_id": "S1"}
        target = {"field": "translation", "quantity_id": "T0"}
        finding = {"category": "number", "severity": "major", **target,
                   "explanation": "The water quantity changed.", "confidence": "high", "uncertainty": "",
                   "evidence": [source, target, {"field": "source", "quote": "25 ml water"}]}
        raw = {"verdict": "findings", "explanation": "Compared quantities.", "findings": [finding]}
        result, flags = finalize_judgment(parse_judgment(raw), item)
        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(item["translation"][slice(*flags[0]["span"])], "३५")
        finding["quantity_id"] = "T9"
        finding["evidence"] = [source, {"field": "translation", "quantity_id": "T9"}]
        self.assertEqual(finalize_judgment(parse_judgment(raw), item)[0]["status"], "inconclusive")

    def test_equal_unit_conversion_is_not_reported_as_a_change(self):
        item = {"source": "Mass 1 kg", "translation": "Mass 1000 g"}
        source = {"field": "source", "quantity_id": "S0"}
        target = {"field": "translation", "quantity_id": "T0"}
        finding = {"category": "number", "severity": "major", **target,
                   "explanation": "The quantity changed.", "confidence": "high", "uncertainty": "",
                   "evidence": [source, target, {"field": "source", "quote": "Mass 1 kg"}]}
        raw = {"verdict": "findings", "explanation": "Compared quantities.", "findings": [finding]}
        result, flags = finalize_judgment(parse_judgment(raw), item)
        self.assertEqual((result["status"], flags), ("inconclusive", []))

    def test_number_span_without_quantity_identifiers_is_inconclusive(self):
        item = {"source": "Use 12 g salt and 25 ml water.", "translation": "पानी ३५ ml और नमक १२ g"}
        source = item["source"].index("25")
        target = item["translation"].index("३५")
        result, flags = finalize_judgment(judgment([source, source + 2], [target, target + 2]), item)
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(flags)

    def test_equivalent_digits_bad_spans_and_uncertainty_abstain(self):
        item = {"source": "Use 25 units.", "translation": "२५ units"}
        for entry in (judgment([4, 6], [0, 2]), judgment([4, 6], [0, 3]),
                      judgment([4, 6], [0, 99]), judgment([4, 6], [0, 2], confidence="low")):
            result, flags = finalize_judgment(entry, item)
            self.assertEqual(result["status"], "inconclusive")
            self.assertEqual(flags, [])

    def test_clean_and_ambiguous_verdicts(self):
        item = {"source": "Use 4 g and 4 g.", "translation": "४ g and ४ g"}
        for verdict, status in (("clean", "evaluated"), ("uncertain", "inconclusive")):
            raw = {"verdict": verdict, "explanation": "Context reviewed.", "findings": []}
            result, flags = finalize_judgment(parse_judgment(raw), item)
            self.assertEqual(result["status"], status)
            self.assertFalse(flags)


if __name__ == "__main__":
    unittest.main()
