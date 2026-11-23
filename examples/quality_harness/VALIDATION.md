> These are limited smoke tests, not a full provider or accuracy validation. Historical runs are identified below.

# Provider smoke validation

The later compact-judge smoke used two previously inspected Tamil items with Groq `openai/gpt-oss-120b`, a ₹1 judge cap and ₹0 Bodhan cap. Three judge attempts succeeded, with estimated judge spend ₹0.11211 and zero Bodhan calls. The compact response on the changed item had valid token IDs, but the finding remained inconclusive because the units were not comparable. Both semantic judgments were inconclusive. All four stable reports matched in a network-blocked replay with zero provider attempts. This tiny diagnostic is not an accuracy measurement. Groq account rate limits and available credit can differ from published rates.

The one-item Hindi run used the installed package and exported graph via `python -I -m bodhan_eval` with `--generate --provider-smoke --max-cost-inr 1`. The assignment environment settings were absent. Forward translation was unavailable, and the dependent synthesis, back-translation, transcription, and structured judgment did not dispatch. The run recorded zero provider calls and ₹0 incremental cost. Actual hosted responses remain unverified.

The current harness export was imported into local Langflow 1.13.2. The browser rendered all five nodes and four connectors. Dataset execution was tested through the headless CLI. Execution from the editor was not tested.

Offline tests use the hosted adapter with mocked HTTP responses. They verify the five request shapes, selected usage fields, WAV parsing, unknown judge cost, and cap admission. A fresh process with network calls blocked replays five cache hits with zero calls and byte-identical reports. The supplied-output run preserves both the translation and WAV bytes when generation is omitted.

## Cookbook `make check` reproduction

Reproduced on a clean detached upstream checkout of `ae51852960911cc78bd71401842308ecc5217510`, using the pinned Cookbook dev dependencies already installed.

```bash
git checkout --detach ae51852960911cc78bd71401842308ecc5217510
PIP_NO_INDEX=1 make check
```

Failure excerpt, with local checkout path prefixes omitted:

```text
tests/test_checks.py::test_all_shipped_recipes_pass FAILED
tests/test_checks.py:171: in test_all_shipped_recipes_pass
    assert not has_errors(findings), [str(f) for f in findings]
[WARNING] structure examples/AndroidSpeak/README.md: README has no '## Run' section
[ERROR  ] structure examples/AndroidSpeak: no notebook or source file found (.ipynb/.py/.js/.ts)
[ERROR  ] structure examples/AndroidSpeak: .env.example is missing
[ERROR  ] structure examples/AndroidSpeak/.gitignore: .gitignore does not ignore .env
1 failed, 31 passed
make: *** [test] Error 1
Exit status: 2
```

The full local log is `outputs/validation/androidspeak-make-check.txt`.
