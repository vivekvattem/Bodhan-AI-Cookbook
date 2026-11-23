# Submission preparation

This is a review draft. The user will record the video, open the PR and send the submission email. No claims below replace the assignment's requirements.

## Harness design

The [architecture](ARCHITECTURE.md) contains the data flow and conditional paths. The CLI runs the exported LFX graph, which delegates checks, provider clients, caching and budget accounting to Python modules. Only Bodhan models handle translation, synthesis and transcription. A general LLM judges existing text and cannot generate or correct Indic text. Human labels join only after detection.

## Results and calibration

The saved sample report is a previously inspected diagnostic, not a held-out benchmark. On the 120-item programmatically injected development set, the prior deterministic-only run identified 12/24 number changes, with 0/24 clean controls flagged and 12/12 detected number spans overlapping the injected edits. It evaluated 12 number checks and left 108 inconclusive. The sample's semantic judge was unavailable.

The [human calibration file](data/human_calibration.jsonl) has 12 reviewed items across two source groups. In the saved run, number errors were detected 2/2, formatting 1/1, omission 0/2, and term/entity 0/1; no flags were raised on 6 clean controls. These small denominators and unavailable judge checks do not justify threshold selection. Other labels remain programmatic and are not human-reviewed.

A separate two-item, previously inspected Tamil diagnostic used Groq `openai/gpt-oss-120b` and no new Bodhan calls. The clean item had no flags; the compact judge returned a structured mismatch for the corrupted item, but Python kept it inconclusive because the units could not be compared. Both semantic judgments were inconclusive. Three judge calls cost an estimated ₹0.11211 at the configured planning exchange rate; this small diagnostic is not an accuracy measurement.

The subsequent [120-item text run](sample_report/live-evaluation/summary.md) completed 120 back-translations and had 70 evaluated, 36 inconclusive and 14 unavailable semantic judgments. On previously inspected programmatic labels, it flagged 12/24 number, 10/24 omission, 2/24 term/entity and 24/24 format cases, with 1/24 clean controls flagged. The human-reviewed subset contains two sources and remains too small for threshold tuning. This run's Groq failed-call charges remain unknown.

## Recorded demo script — five minutes or less

1. Explain the goal: identify translation and speech issues while surfacing uncertainty.
2. Show the dataset contract and the English source, language and supplied output fields. Keep credentials off screen.
3. Run the documented CLI against the demo using an explicit budget, or show the recorded sample report if provider access is unavailable.
4. Open `summary.md` and one item in `items.jsonl`; point out the original-field span and a skipped or inconclusive check.
5. Show calibration support and cache-only replay. State that the inspected demo is not a held-out benchmark.
6. Explain remaining gaps: human review, provider availability, speech benchmarks and missing checks.

## PR draft

**Title:** `[Agentic Eng Assignment] Quality harness — <your name>`

- **Summary:** CLI and Langflow flow for translation and speech-quality checks, evidence and reports.
- **How to run:** Follow [README](README.md), using the six assignment environment settings and a budget.
- **Architecture and harness design:** See [architecture](ARCHITECTURE.md) and [README](README.md).
- **LLM use:** Semantic and compact residual reviews; evidence validated in Python. The live smoke used Groq `openai/gpt-oss-120b`; judge consistency is not established.
- **Bodhan models:** Indic-Translate, Indic-Speak and Indic-Transcribe, hosted calls.
- **Results:** Use the scoped values above and [sample report](sample_report/summary.md); include check coverage.
- **Demo video:** Add the user-recorded accessible link.
- **Trade-offs:** Sparse human calibration, previously inspected development data, inconclusive checks, fallible transcription and missing reference/wrong-language checks.
- **Cost and runtime:** See the [200-row warm-cache speech diagnostic](sample_report/speech-benchmark/README.md). Recognition and the judge were incomplete, so the full end-to-end 200-item runtime target remains unverified. Do not equate cached replay cost with live cost.
- **AI assistance:** Describe the actual tools used and their contribution.

## Remaining verification

Verify setup from a clean clone with the pinned dependencies, reproduce reports and replay, validate provider credentials and budget, run a fresh human-reviewed source-disjoint evaluation, perform an independent speech listening review, and measure complete end-to-end runtime/cost with speech. The assignment requires generated audio to stay out of Git and be linked from the README; an accessible hosted-audio link is still pending. Root `make check` has reproduced an unrelated AndroidSpeak recipe structure failure; retain the exact test output and commit used when reporting it. Check the final commit for secrets and generated audio before the user submits.
