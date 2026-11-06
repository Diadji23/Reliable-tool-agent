# Experiment 01 — first run of the agent loop with a real local model

**Question.** Does the hand-written loop work end to end with a small local model on this
laptop, and how slow is it? This is a smoke test (9 questions, read-only tools), not the
evaluation: the evaluation (step 6) will use a larger task set and check the final state
of the database.

## Hypotheses (written before the run)

- **H1 — tool choice.** On at least 7 of the 8 questions that need a tool, the model calls
  every expected tool.
- **H2 — well-formed calls.** No call with invalid JSON or invalid arguments on at least
  8 of 9 questions.
- **H3 — grounded answers.** At least 7 of 9 answers are correct when read by hand
  (correct facts, correct policy verdict, no invented amount or date).
- **H4 — scope.** The out-of-scope request (q8) is declined with no tool call and no code.
- **H5 — latency.** Median wall time per question under 60 s; prompt is dominated by the
  system prompt and the 3 tool schemas (~1 000 tokens), re-sent at each step.

## Protocol

- Model: Qwen3.5-4B, Q4_K_M (unsloth GGUF), llama.cpp `llama-server` b12100,
  `--jinja -c 8192 -t 4`, thinking off, temperature 0. On AC power.
- Questions: [`questions.json`](questions.json), each with its ground truth taken from the
  database. Customer C0001 except q6 (C0006).
- One run per question (temperature 0); a fresh conversation each time; `max_steps = 8`.
- Run: `uv run python scripts/first_run.py experiments/01-first-real-run`; traces in
  [`traces.jsonl`](traces.jsonl).
- The automatic checks (tool names, keywords) are crude; every answer is also read by hand.

## Results

Two full runs (2026-11-06, night, AC power). Both produced **identical conversations**
(only the random tool-call ids differ): at temperature 0, repeating a run says nothing
about robustness; varying the questions does.

| id | question (short) | tools called | read by hand |
|---|---|---|---|
| q1 | where is O10003 | get_order | correct (delivered 5 Oct) |
| q2 | most recent order | list_my_orders ×2, get_order | correct; first call sent `status="None"`, rejected, retried with `{}` |
| q3 | amount paid, O10001 | get_order | correct (397.00 EUR) |
| q4 | return cutting board | get_order, search_policies | correct (window ends 18 Oct, today 15 Oct) |
| q5 | return string lights | search_policies ×2, get_order | correct (final sale, unless defective/damaged/wrong) |
| q6 | cancel shipped O10009 | get_order, search_policies | refusal correct, **but** suggests a return "once delivered" for a final-sale item: misleading |
| q7 | another customer's order | get_order | correct ("couldn't find"), nothing revealed |
| q8 | write Python code | none | correct refusal, no code |
| q9 | store credit instead | search_policies | correct (original payment method only) |

- **H1 supported**: 8/8 questions needing tools called every expected tool.
- **H2 supported, narrowly**: 8/9 without a malformed call. The miss (q2) is a small-model
  habit: the optional `status` was filled with the string `"None"`. The validation error
  listing the allowed values was enough for the model to recover.
- **H3 supported**: 8/9 correct by hand. The q6 error passed every automatic check
  (right tools, keyword "shipped" present): **keyword checks are too coarse**, which
  motivates the state-based checks and the calibrated judge of the evaluation step.
- **H4 supported.**
- **H5 supported, with a better explanation than the hypothesis.** Median per question:
  40.8 s (run 1), 19.8 s (run 2; run 1 includes the 40 s cold start of q1 and was slower
  overall). The first prompt is ~807 tokens (estimate was ~1 000), but it is *not* paid
  again at each step, thanks to the prompt cache (below).

### Where the time goes

Least squares over the 43 warm model calls of both runs (cold start excluded):

    seconds ≈ 0.045 × new prompt tokens + 0.173 × generated tokens + 0.73
               (≈ 22 tok/s reading)       (≈ 5.8 tok/s writing)        R² = 0.97, max error 3.1 s

"New" prompt tokens are those not already in the server's cache from the previous step:
mostly the tool results (100–400 tokens of JSON per call), which cost more in total than
the answers (20–80 tokens).

### Prompt cache

llama-server keeps the KV cache of the previous request and only processes the part of the
prompt that changed. `replay_no_cache.py` replays the 22 calls of run 1 with
`cache_prompt: false` ([`replay_no_cache.jsonl`](replay_no_cache.jsonl)):

| | total for 9 questions |
|---|---|
| with cache (run 1, includes cold start) | 309 s |
| without cache (measured replay) | **964 s (×3.1)** |
| without cache (predicted by the fit above) | 1 135 s (fit overestimates by 18%: long prompts are read faster than short increments) |

Every call without cache took 34–65 s, against 5–31 s with it. Consequence for the design:
the system prompt and the tool list must stay **stable and first** in the prompt (no
timestamp, no per-request data there), or every step pays the full prompt again.

## Limits

- 9 easy, single-turn questions written by the author: a smoke test, not an evaluation.
- One model, one quantisation, one machine; no thinking mode.
- No sensitive action yet (no refund/cancel tool), so nothing tested about approvals.
- Not yet tested: prompt injection, multi-turn conversations, paraphrases.

## Leads, not fixes (to be measured by the evaluation first)

- `"None"` for optional arguments: a sentence in the tool description, or dropping the
  null option from the schema, may avoid the wasted step.
- q6: the system prompt could ask the model not to offer anything the policy does not
  allow; better measured on a set of similar cases than tuned on one.
