# Bodhan Quality Harness

The Python CLI runs a five-stage Langflow export headlessly to evaluate supplied or generated translation and speech outputs. It reports evidence, check coverage, and estimated model costs. The [architecture guide](ARCHITECTURE.md) explains the current design. The [implementation plan](PLAN.md) covers later stages.

## Setup

The recipe was exercised on Python 3.11.4 with LFX 1.13.2. From `examples/quality_harness/`, install the package and pinned direct dependencies in a virtual environment.

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install .
source .venv/bin/activate
```

### Local provider settings

From the Cookbook checkout root, `cd examples/quality_harness`. Keep `.env` in that recipe directory and run the commands below from there. On a fresh checkout, create `.env` by copying `.env.example` once. Paste each value after its `=` in `.env` when it is available.

| Setting | Value |
| --- | --- |
| `BODHAN_TRANSLATE_KEY` | Your Indic-Translate key from the Bodhan console |
| `BODHAN_SPEAK_KEY` | Your Indic-Speak key from the Bodhan console |
| `BODHAN_TRANSCRIBE_KEY` | Your Indic-Transcribe key from the Bodhan console |
| `LLM_BASE_URL` | The OpenAI-compatible API base URL from your judge provider, including its API version prefix when applicable |
| `LLM_MODEL` | The judge model ID from that provider |
| `LLM_API_KEY` | Your API key from that judge provider |

The CLI reads `.env` from the current working directory, including with `python -I -m bodhan_eval`. Variables already set in the shell take precedence. Missing credentials are reported on affected checks. The generic `BODHAN_API_KEY` entry in `.env.example` is a Cookbook validator placeholder. [Bodhan API reference](https://console.bodhan.ai/api-docs/)

## Run

```bash
python -m bodhan_eval \
  --dataset path/to/data.jsonl \
  --langs hi,ta,bn \
  --out ./report \
  [--generate] [--baseline ./previous_report] [--calibration labels.jsonl] [--max-cost-inr 50]
```

Supply `id`, English `source`, and `lang` on each UTF-8 JSONL row. Optional fields are `translation`, `reference`, `audio`, and `labels`. Resolve relative audio paths against the dataset directory. `--generate` fills missing translations and audio and preserves supplied outputs. Unreadable supplied audio remains unavailable. `--calibration` reads records shaped like `{"id":"example","lang":"hi","labels":[]}` and gives those annotations precedence for evaluation. An empty label array marks an annotated clean item. An absent array leaves the item unlabelled. `--baseline` compares against `items.jsonl` in a previous report directory.

### One-item provider smoke test

From the activated environment, run the production CLI with the six assignment settings loaded from the environment or local `.env`.

```bash
python -I -m bodhan_eval --dataset sample_data/provider_smoke.jsonl --langs hi \
  --out outputs/provider-smoke/report --generate --provider-smoke
```

`--provider-smoke` accepts one selected non-English item. The graph generates a translation and WAV, back-translates the translation, transcribes the WAV, and requests a three-value JSON verdict. The verdict appears under `provider_smoke_judge` and has no effect on findings. Check the five provider statuses in `items.jsonl`. Semantic translation and basic speech-signal and transcript-alignment checks are implemented; inspect their individual statuses and limitations. `report.runtime.json` records the requested model, allowlisted settings, call outcomes and latency, token usage when returned, parsed WAV metadata, and per-request cost estimates or an unknown-cost reason.

The uncapped form makes at most one unpriced judge request with a 64-token output limit. Add `--max-cost-inr 1` to apply a total cap. An uncached judge call without a known tariff is then unavailable. A complete warm replay uses `--cache-only` and an output under the same parent directory.

```bash
python -I -m bodhan_eval --dataset sample_data/provider_smoke.jsonl --langs hi \
  --out outputs/provider-smoke/replay --generate --provider-smoke --cache-only
```

Compare the four report files byte for byte and check the replay telemetry for zero provider calls and zero incremental spend. To check supplied outputs, use a copy of the row with the generated translation and local WAV path and omit `--generate`. Generation checks should show `skipped` and the supplied artifacts should be preserved.

The one-item test suite exercises the production HTTP adapter with mocked responses and verifies replay in an isolated process with network access blocked. The production CLI ran without credentials and reported all five operations unavailable. Bodhan operations and a small judge smoke were exercised separately; see the validation notes for scope and limitations. See [validation notes](VALIDATION.md).

### Reproduce the offline fixture

```bash
mkdir -p outputs/fixture
cp sample_data/milestone.jsonl outputs/fixture/milestone.jsonl
python tests/offline/fixture_transport.py outputs/fixture/audio/fixture.wav

python tests/offline/run_fixture.py \
  --dataset outputs/fixture/milestone.jsonl \
  --langs hi,ta,bn,ur \
  --out outputs/fixture/report \
  --generate --max-cost-inr 50
```

The test-only runner explicitly imports and passes `FixtureTransport`. It supplies one synthetic forward translation, digit-folded back-translation markers, short silent WAVs, a fixed transcript marker, and a fixed judge verdict. The supplied WAV is generated locally for this test. The production CLI always selects the hosted adapter. Fixture reports are labelled `fixture-stub` and exercise the harness with synthetic evidence and costs.

### Run the tests and the original proof

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s proof -v
python proof/run.py '२५' '25'
python proof/run.py '३५' '25'
```

The proof compares `25` with `25` and `35` with `25` after digit folding. Its saved export and the harness export use the same pinned Python execution interface. [LFX headless execution](https://docs.langflow.org/lfx-overview) · [Pinned Python execution interface](https://github.com/langflow-ai/langflow/blob/v1.13.2/src/lfx/src/lfx/load/load.py)

## How it works

The CLI validates rows independently and passes allowlisted inputs to the exported Langflow graph. Python modules prepare outputs, run deterministic number and formatting checks, gather Bodhan back-translation and transcription evidence, request semantic judgment, and validate original-text spans. Labels are joined after detection. Audio checks measure silence and clipping; transcript alignment supplies fallible evidence rather than proof of a speech error.

The number check folds Unicode decimal digits, retains original code-point spans, and promotes a single residual pair only under its supported safeguards. Ambiguous correspondence, unsupported notation, and incompatible units remain inconclusive. Supported unit conversions are compared. A compact residual-ID judge is conditionally called for unresolved, uniquely identifiable quantity residuals. Python validates its IDs and evidence; ambiguous or unsupported pairings remain inconclusive.

The 120-item [text demo](data/README.md) is previously inspected development data derived from eight English sources. [Human calibration labels](data/human_calibration.jsonl) cover 12 reviewed items from two sources; pass the file with `--calibration`. Its results are descriptive and do not justify threshold tuning. A [sample report](sample_report/summary.md) is included with its [calibration details](sample_report/calibration.json); its judge was unavailable. It is not a held-out accuracy measurement.

A later [live text evaluation](sample_report/live-evaluation/summary.md) used cached and new Bodhan back-translations plus Groq judgment. It records incomplete checks and unknown failed-attempt billing; these previously inspected data do not establish held-out accuracy.

The [speech runtime and coverage diagnostic](sample_report/speech-benchmark/README.md) describes a 200-row warm-cache run and its rate-limited transcription retry. The workload contains related repeats and does not establish a complete 200-item end-to-end runtime target.

Reports include `summary.md`, `summary.json`, `items.jsonl`, `calibration.json`, and `diff.md` when a baseline is supplied. Exact requests are cached, with bounded retries and explicit budget admission. Invocation telemetry is stored separately from stable reports. Use `--cache-only --replay-from <original-report>` with the same cache and detection settings to reproduce a saved run.

## Limitations

Compact and semantic judge checks can be unavailable or inconclusive; basic speech alignment is fallible and needs independent listening review. Reference scoring, wrong-language detection, speech formats beyond short PCM WAV, threshold tuning, and the 200-item speech-inclusive runtime target remain unverified or pending. The human-reviewed calibration subset is too small for reliable quality estimates.

The end-to-end tests also run the exact production entry point with missing credentials. Run the Cookbook checks from a clean checkout root because its security scanner inspects working-tree files, including ignored `.env` files.

```bash
python scripts/validate_recipe.py examples/quality_harness
make check
```

Submission preparation, human-review limitations and a video script are in [SUBMISSION.md](SUBMISSION.md).

Generated audio is kept out of Git. An accessible audio link for the submission remains to be provided; the video link also remains pending.
