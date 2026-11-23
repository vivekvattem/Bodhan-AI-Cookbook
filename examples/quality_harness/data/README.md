# Bodhan demo data

This is development data for the quality harness, not a validated benchmark.
`demo.jsonl` contains only detector inputs. `ground_truth.jsonl` contains edit
annotations. Never join the two before detection. The Langflow graph receives
only the dataset loader's allowed content fields.

## Provenance and review

`sources.jsonl` contains eight original synthetic English educational passages
written for this recipe with AI coding assistance on 2026-11-22. No external
passage or private material was copied. The topics are measurement, plant
growth, geometry, states of matter, maps, algorithms, chronology, and ecology.
They may be redistributed with this recipe under the Cookbook's license.

Only Bodhan Indic-Translate produces the Hindi, Tamil, and Bengali paragraphs
in `base_translations.jsonl`. Each English paragraph is translated separately.
The source and target share a literal table or math scaffold. Its code and math
identifiers are intentionally language independent; they are not untranslated
prose. The preparation tool does not call a general LLM, recognition, or TTS.

Each of the 24 bases has five variants: unchanged control, paragraph omission,
one changed decimal digit, literal technical-identifier substitution, and
deleted table or math marker. That yields 120 text-only items, 40 per language.
The term swaps exercise symbolic terms, not general vocabulary or all entities.
All Indic edits are mechanical; an LLM never writes or corrects Indic text.

The separate `human_calibration.jsonl` contains 12 manually reviewed calibration items across two source groups. The programmatic labels remain unreviewed unless explicitly marked otherwise. A clean control means no injected edit,
not a certified correct Bodhan translation. Ground truth records known edits
relative to a base, with `human_review: pending` and
`base_translation_verified: false`. Review the base translation, educational
meaning, category, severity, and linguistic effect before using these records
as valid quality labels. Programmatic severity is provisional. Do not report
accuracy from raw flag counts or describe these as human labels.

## Files and splits

- `sources.jsonl`: original English paragraphs, topic, literal markup and term pairs.
- `base_translations.jsonl`: Bodhan paragraphs before edits; no API credentials.
- `demo.jsonl`: all 120 detection records; no labels or injection metadata.
- `ground_truth.jsonl`: operator, category, source item, original edit span/text,
  replacement and output span, annotation provenance, and base fingerprint.
- `calibration.jsonl` and `evaluation.jsonl`: detector inputs split by English
  source group, across all languages and derived variants.
- `calibration_labels.jsonl` and `evaluation_labels.jsonl`: separate annotations.
- `manifest.json`: seed, counts, split sizes, review state, and file fingerprints.

The seed is `20261122`. Two source groups (30 items) are reserved for calibration
and six (90 items) for evaluation. All variants and languages of one English
source stay together. No threshold was tuned on either partition. The earlier
four-clip speech diagnostic is separate development data, not part of this set.

Spans are half-open Unicode code-point offsets. `edit.original_span` selects the
base translation and `edit.output_span` selects its replacement (an empty
anchor for deletion). Omission labels select the paired original English
paragraph. Deleted formatting markers select `clean_translation`, not a
nonexistent character in the corrupted output. A later localization evaluator
must use the edit mapping and support zero-length deletion anchors correctly.

## Reproduce and evaluate

Run from `examples/quality_harness/` in the installed pinned environment, with
the assignment keys in the local ignored `.env` or existing environment.

```bash
# Paid preparation: only Indic-Translate, successful exact requests cached.
python -m bodhan_eval.demo generate --sources data/sources.jsonl \
  --out outputs/demo-base-20261122 --max-cost-inr 1

# Offline build from the saved Bodhan bases. No models are called.
python -m bodhan_eval.demo build --sources data/sources.jsonl \
  --bases outputs/demo-base-20261122/base_translations.jsonl --out data \
  --seed 20261122

# Actual required CLI and exported Langflow graph. Supplied text is preserved.
python -m bodhan_eval --dataset data/demo.jsonl --langs hi,ta,bn \
  --out outputs/demo-report-20261122 --max-cost-inr 3
```

The preparation command pauses four seconds after each live dispatch and stops
on the first HTTP 429, so an explicit later resume reuses completed paragraphs.
Preserve partial preparation results before resuming. An initial unpaced
attempt hit rate limits; this session reserved up to INR 1 for that attempt,
INR 1 for the paced resume, and INR 3 for evaluation, an aggregate INR 5 cap.
Failed-call billing is unknown and retained conservatively in the reservation.
Pricing estimates are not account statements. Actual attempts, costs, coverage
and unavailable checks should be recorded in the generated runtime and report files.

The report contract remains `summary.md`, `summary.json`, `items.jsonl`, and
`calibration.json`, plus `diff.md` with `--baseline`. Variable telemetry is an
adjacent `.runtime.json`. Pass `--calibration data/human_calibration.jsonl` to report descriptive metrics against the human-reviewed subset; its small and previously inspected sample does not support threshold tuning. Rate limits, failures, or a cap may leave model checks unavailable;
they do not establish clean results. Reference scores and wrong-language/script
checks remain pending where applicable. This text-only run skips speech checks.

For an offline replay after cache population:

```bash
python -m bodhan_eval --dataset data/demo.jsonl --langs hi,ta,bn \
  --out outputs/demo-replay-20261122 --max-cost-inr 3 --cache-only
```

Stable report equality requires the same run identity and saved observations. Use `--cache-only --replay-from <original-report>` and compare files; inspect coverage rather than assume completeness.

## Optional audio

No generated audio is included in this directory or Git. To obtain it, select
one clean record into an ignored JSONL under `outputs/`, then use the same
`python -m bodhan_eval` command with `--generate` and an explicit cost cap.
That preserves the supplied translation and synthesizes missing audio only
through Indic-Speak; recognition uses Indic-Transcribe. Use short samples and
inspect generation/check statuses. Generated WAV files and caches live under
the output parent's ignored `.bodhan-eval/`. This task does not purchase audio.

`bodhan_eval.inject.audio_variant` also supports deterministic PCM truncation,
segment repetition, and inserted silence. It returns separate signal-edit
metadata with source item, effective original and output intervals, and a
fingerprint. It does not label words, complete phrases, or word timestamps.
Human listening and transcription are required for linguistic speech labels.
