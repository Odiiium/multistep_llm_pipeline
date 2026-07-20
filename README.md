# Multi-step LLM pipeline

A pipeline for processing user messages: each message goes through five steps and becomes structured JSON with an answer and a quality score.
The main focus of the project is **resilience**: retries with error classification, a fallback ladder, validation of model responses, structured logging, and result persistence on crash.

---

## Structure

```
itgrind_transformers/
    main.py                     batch run over a file
    multistep_llm_pipeline.py   pipeline + interactive mode + assembly
    llm_api.py                  API clients
    data_models.py              pydantic models and types
    promts.py                   prompts, schemas, fallback prompts
    llm_core/
        errors.py       TRANSIENT / BAD_OUTPUT / FATAL classification
        retry.py        backoff + jitter + Retry-After
        validation.py   five checks, sanitizer, length, strict schemas
        logs.py         JSONL logger
        guards.py       the single model call site, ladder L0-L2
        policies.py     per-step policy
        fallback.py     deterministic L3 fallbacks
    tests/              offline checks on fake clients
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r itgrind_transformers/requirements.txt
```

## API keys

Two ways; the command-line argument takes precedence.

**Via `.env`** in the `itgrind_transformers/` folder:

```
DEEP_INFRA_KEY=your_main_model_key
OPEN_AI_KEY=your_simple_model_key
```

The file is already in `.gitignore`.

**Via arguments:**

```bash
python itgrind_transformers/main.py --api-key sk-... --simple-api-key sk-...
```

### Models and endpoints

The pipeline has two "heads":

- **main** (`--api-key` / `--model` / `--base-url`) — `extract_sense`, `final_answer`, `judge`;
- **simple** (`--simple-api-key` / `--simple-model` / `--simple-base-url`) — `intent` and `extract_fields`.

The endpoint is set independently for each, so both heads can live at the same provider or at different ones.

**One model for everything:**

```bash
python itgrind_transformers/main.py --api-key sk-... --model llama-3.3-70b
```

**Two models from the same provider** — the endpoint is inherited from the main head, no need to repeat it:

```bash
# both on DeepInfra
python itgrind_transformers/main.py \
  --api-key sk-deepinfra... \
  --model        deepseek-ai/DeepSeek-V4-Flash \
  --simple-model meta-llama/Meta-Llama-3.1-8B-Instruct

# both on OpenAI
python itgrind_transformers/main.py \
  --api-key sk-openai... --base-url https://api.openai.com/v1 \
  --model gpt-4o --simple-model gpt-4o-mini
```

**Two models from different providers** — two keys; the simple head defaults to OpenAI:

```bash
python itgrind_transformers/main.py \
  --api-key        sk-deepinfra... --model        deepseek-ai/DeepSeek-V4-Flash \
  --simple-api-key sk-openai...    --simple-model gpt-4o-mini
```

**Different providers, neither of them OpenAI** — the simple head's endpoint is set explicitly:

```bash
python itgrind_transformers/main.py \
  --api-key        sk-a... --model        model-a --base-url        https://provider-a/v1 \
  --simple-api-key sk-b... --simple-model model-b --simple-base-url https://provider-b/v1
```

How the simple head is resolved:

| What is given | Key | Model | Endpoint |
|---|---|---|---|
| Only `--api-key` | same | `--model` | `--base-url` |
| `--simple-api-key` matches `--api-key` | same | `--simple-model` or `--model` | `--simple-base-url` or `--base-url` |
| `--simple-api-key` differs | its own | `--simple-model` or `gpt-4o-mini` | `--simple-base-url` or OpenAI |
| Nothing is given | env `OPEN_AI_KEY` | `gpt-4o-mini` | OpenAI |

> `OPEN_AI_KEY` from `.env` is picked up **only when `--api-key` is not passed**. Otherwise an explicit argument would silently split the pipeline across two providers.

Full list of arguments:

| Argument | Default | Purpose |
|---|---|---|
| `--api-key` | env `DEEP_INFRA_KEY` | Key for the main model |
| `--model` | `deepseek-ai/DeepSeek-V4-Flash` | Main model |
| `--base-url` | `https://api.deepinfra.com/v1/openai` | Endpoint of the main model |
| `--simple-api-key` | env `OPEN_AI_KEY`, if `--api-key` is absent | Key for the simple model |
| `--simple-model` | see the table above | Model for intent and field extraction |
| `--simple-base-url` | see the table above | Endpoint of the simple model |
| `--lines` | `10` | How many lines to read from the input file, `-1` for all |
| `--input` | `llm_pipeline_results/input.txt` | Input file, one message per line |
| `--output` | `llm_pipeline_results/output.txt` | Where to write the results |

> The `strict` flag for structured outputs is enabled automatically only when the simple model runs on an OpenAI endpoint. On a third-party provider it is turned off, otherwise the request would return 400.

> Configuration errors (401, 403, 404, 400 — wrong key, non-existent model, foreign endpoint) stop the run immediately with a clear message. They deliberately do not degrade into a fallback: otherwise the pipeline would "successfully" finish on deterministic stubs and return a silently worse result.

---

## Running

### Batch mode

Reads a file of lines and writes a JSON array of results.

```bash
python itgrind_transformers/main.py --lines 5
python itgrind_transformers/main.py --input my_messages.txt --output result.json --lines -1
```

### Interactive mode

Takes messages from the console one at a time and prints the result JSON.

```bash
python itgrind_transformers/multistep_llm_pipeline.py
```

To quit use `exit`, `quit`, or Ctrl+D. Each run creates its own file `llm_sessions/session_YYYYMMDD_HHMMSS.jsonl`, where results are appended line by line.

### Tests

Fully offline, no network and no tokens:

```bash
python itgrind_transformers/tests/test_resilience.py   # 91 checks
python itgrind_transformers/tests/test_durability.py   # 23 checks
```

---

## Input and output examples

### Input

One message per line, `llm_pipeline_results/input.txt`:

```
The new update is great, but it would be useful to have more customization options.
I think your service needs better documentation for new users.
The application crashes every time I try to upload a file.
```

### Output

`output.txt` is a JSON array. A single element:

```json
{
  "question_index": 2,
  "start_question": "The application crashes every time I try to upload a file.",
  "summary": "The application crashes every time I try to upload a file.",
  "category": "support",
  "sentiment": "negative",
  "key_points": [
    "The application crashes consistently during the file upload process.",
    "The issue is urgent and has a negative impact on user experience.",
    "The root cause could be related to file size, file type, or a bug in the upload module."
  ],
  "final_answer": "I'm sorry you're experiencing this crash during file uploads. Let's try a few steps to resolve it. First, check if the file is under 25 MB and in a common format like PDF or JPEG. If it is, try clearing your browser cache or restarting the app.",
  "judge_result": {
    "passed": true,
    "score": 10,
    "issues": []
  },
  "degraded_steps": [],
  "fallback_levels": {
    "extract_sense": 0,
    "intent": 0,
    "extract_fields": 0,
    "final_answer": 0,
    "judge": 0
  }
}
```
`degraded_steps` and `fallback_levels` show at what cost the result was obtained. An example of a line where sense extraction failed at every level and the deterministic fallback kicked in:

```json
"degraded_steps": ["extract_sense"],
"fallback_levels": {"extract_sense": 3, "intent": 0, "extract_fields": 0, "final_answer": 0, "judge": 0}
```

Internally the extraction step uses a schema specific to each intent — `support` has `problem` / `product` / `error_code`, `sales` has `interester_product` / `budget` / `items_count`, `feedback` has `feedback_type` / `subject` / `feature` / `suggestion`, and so on. These fields are context for the answer step; only `summary` and `sentiment` are lifted into the result, so the output shape stays the same for every intent.

---

## Run artifacts

| Path | What is inside |
|---|---|
| `llm_logs/pipeline.jsonl` | One line per model call, including retries. The source of metrics |
| `llm_logs/failures.jsonl` | Raw text of rejected responses with the reason — material for fixing prompts |
| `llm_pipeline_results/output_partial.jsonl` | Incremental write: on a crash, finished lines stay on disk |
| `llm_sessions/session_*.jsonl` | Results of interactive sessions |

A call log record:

```json
{
  "ts": "2026-07-19T12:41:04.950+00:00",
  "run_id": "777519d8", "trace_id": "e1ce5f28", "line_idx": 1,
  "step": "intent", "model": "gpt-4o-mini", "attempt": 1,
  "fallback_level": 0, "fallback_mode": "normal",
  "outcome": "ok", "temperature": 0.01,
  "finish_reason": "stop", "response_len": 132,
  "sanitized": false, "soft_limit_hit": true,
  "latency_ms": 986, "prompt_tokens": 415, "completion_tokens": 26
}
```

Quick metrics from the log:

```bash
python -c "
import json; from collections import Counter
rows=[json.loads(l) for l in open('itgrind_transformers/llm_logs/pipeline.jsonl')]
print(Counter(r['outcome'] for r in rows))
print('truncated:', sum(1 for r in rows if r.get('finish_reason')=='length'))
print('avg attempts:', sum(r['attempt'] for r in rows)/len(rows))
print('fallback levels:', Counter(r.get('fallback_level') for r in rows))
"
```

---

## What it does

Every input message goes through a chain of five steps:

| Step | What it does | Criticality |
|---|---|---|
| `extract_sense` | Extracts the gist of the message without filler | optional |
| `intent` | Classifies the intent: `support`, `feedback`, `complaint`, `sales`, `general_question` | degradable |
| `extract_fields` | Pulls out structured fields using a schema specific to each intent | degradable |
| `final_answer` | Generates the answer for the user plus exactly 3 key points | **required** |
| `judge` | Checks the answer for contradictions and made-up facts | optional |

The pipeline can run on two different models: a "heavy" one for generation and a "simple" one for classification and field extraction. You can also point everything at a single model.

### Resilience

**Error classification.** Failures fall into three classes with different strategies:

- `TRANSIENT` (network, 429, 5xx, timeout) — retry with exponential backoff and jitter, honouring the `Retry-After` header;
- `BAD_OUTPUT` (HTTP 200, but the response is unusable) — retry with changed conditions;
- `FATAL` (401, 403, 404, 400) — no retries, immediate stop: this is a configuration error, and three attempts would only mask the cause.

**Response validation** — five checks in a fixed order:

```
1. empty / whitespace-only / refusal   -> empty
2. finish_reason == "length"           -> truncated
3. json.loads                          -> invalid_json
4. pydantic                            -> schema
5. length > hard_max                   -> too_long
```

The order matters: the truncation check comes before JSON parsing, otherwise a response cut off by `max_tokens` would always be diagnosed as "invalid JSON".

**Fallback ladder** — on a bad response the step escalates instead of repeating the same request:

| Level | What happens |
|---|---|
| L0 | Regular prompt and temperature |
| L1 | The validation error, the previous bad answer and a targeted hint are added to the conversation; temperature is lowered |
| L2 | The system prompt is replaced with a simplified one containing an example, temperature drops to the floor |
| L3 | Deterministic fallback without calling the LLM |
| L4 | The step is marked `degraded` and the pipeline moves on |

A `judge` failure does not drop a finished answer to the user — only `final_answer` is mandatory.

**Length limits.** The length of prose text is set as a budget in the prompt and checked after generation (soft is a warning, hard is a rejection). Length limits on prose are deliberately kept out of the JSON Schema: the structured-outputs grammar cuts the text exactly at the limit, mid-word.
