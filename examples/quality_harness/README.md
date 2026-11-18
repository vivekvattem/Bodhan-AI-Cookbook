# Bodhan Quality Harness

This milestone evaluates decimal numerals in supplied or generated translations. The Python CLI runs a five-stage Langflow export headlessly and reports evidence, coverage, and model call costs. The [implementation plan](PLAN.md) covers later stages.

## Setup

The recipe was exercised on Python 3.11.4 with LFX 1.13.2. From `examples/quality_harness/`, install the package and pinned direct dependencies in a virtual environment.

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install .
source .venv/bin/activate
```

For hosted checks, set the model-specific variables in the environment or copy `.env.example` to `.env` in the working directory. The CLI uses the three Bodhan keys and the `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` settings. The generic `BODHAN_API_KEY` field is a Cookbook template placeholder. Missing configuration is reported on the affected check. [Bodhan API reference](https://console.bodhan.ai/api-docs/)

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

`--provider-smoke` accepts one selected non-English item. The graph generates a translation and WAV, back-translates the translation, transcribes the WAV, and requests a three-value JSON verdict. The verdict appears under `provider_smoke_judge` and has no effect on findings. Check the five provider statuses in `items.jsonl`. Full semantic and speech checks remain pending. `report.runtime.json` records the requested model, allowlisted settings, call outcomes and latency, token usage when returned, parsed WAV metadata, and per-request cost estimates or an unknown-cost reason.

The uncapped form makes at most one unpriced judge request with a 64-token output limit. Add `--max-cost-inr 1` to apply a total cap. An uncached judge call without a known tariff is then unavailable. A complete warm replay uses `--cache-only` and an output under the same parent directory.

```bash
python -I -m bodhan_eval --dataset sample_data/provider_smoke.jsonl --langs hi \
  --out outputs/provider-smoke/replay --generate --provider-smoke --cache-only
```

Compare the four report files byte for byte and check the replay telemetry for zero provider calls and zero incremental spend. To check supplied outputs, use a copy of the row with the generated translation and local WAV path and omit `--generate`. Generation checks should show `skipped` and the supplied artifacts should be preserved.

The one-item test suite exercises the production HTTP adapter with mocked responses and verifies replay in an isolated process with network access blocked. The production CLI ran without credentials and reported all five operations unavailable. Actual hosted responses and visual-editor import remain unverified. See [validation notes](VALIDATION.md).

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

The CLI validates rows independently and passes only allowlisted content to `src/bodhan_eval/quality_harness.json`. The five nodes prepare outputs, check numerals, gather back-translation and transcription evidence, request a bounded judge verdict for number findings, and validate original spans. The judge verdict is recorded separately. The deterministic candidate supplies the finding and its evidence. Python joins annotations after the graph returns. The export can be regenerated after an editable install with `python tools/export_flow.py`. Visual-editor import is unverified because the editor was unavailable. Headless execution and export validation were tested.

The decimal check folds Unicode decimal digits, compares exact decimal values, and uses half-open Unicode code-point ranges to slice the original texts. A changed value becomes a finding when each text contains one numeral with the same supported explicit unit. Mismatched units, unpaired changes, grouped numbers, fractions, scientific notation, dates, ranges, and attached identifiers are inconclusive. Spelled-out numbers, unit conversion, localized unit words, and semantic correspondence are outside the check. Matching numeric values establish limited numeric coverage.

The reports are `summary.md`, `summary.json`, `items.jsonl`, `calibration.json`, and `diff.md` when a baseline is supplied. Item counts reflect the number check. Separate check counts show unavailable, skipped, and pending model or quality checks. Calibration gives descriptive item-level number counts when eligible labels exist. Threshold tuning and judge calibration are pending. An unavailable current check cannot resolve a previous number finding.

SQLite stores validated responses under exact request hashes in `.bodhan-eval/` beside the report directory. Generated WAVs there are named by content hash. The report retains the original evaluation cost estimate for the evidence used. The adjacent `<out>.runtime.json` records invocation calls, cache hits, elapsed time, and incremental estimated spend. `--max-cost-inr` reserves known Bodhan costs before uncached dispatch and leaves unpriced hosted judge calls unavailable when a cap is supplied. Requests are serial for this milestone. A complete replay uses cached responses at zero incremental cost. Failed provider calls can be attempted on a later invocation. [Bodhan published rates](https://console.bodhan.ai/#pricing)

## Limitations

Live provider access, full translation semantics and speech detectors, audio chunking and formats beyond short PCM WAV, retries and concurrent scheduling, threshold tuning, and the 100-item demo remain pending. The judge supports number review and the separate connectivity check. Long translation responses may hit the milestone’s output cap and are reported as unavailable when the provider indicates truncation.

The end-to-end tests also run the exact production entry point with missing credentials. To run the Cookbook checks from a Cookbook checkout root, use:

```bash
python scripts/validate_recipe.py examples/quality_harness
make check
```
