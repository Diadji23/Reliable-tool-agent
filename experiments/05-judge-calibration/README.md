# Experiment 05 — calibrating the LLM judge

**Question.** Does the judge ([`judge.py`](../../src/reliable_tool_agent/judge.py)) agree
with a human reading of the agent's words? A judge is a model too: until it is measured
against hand labels, its verdicts are worth no more than the agent's.

**Data.** [`evaluation/judge_labels.json`](../../evaluation/judge_labels.json): 74 hand
verdicts on the words of whole conversations. Two sets, kept apart:

- **tuning set**: experiment 03, before and after the fix (46 conversations, 13 fail).
  The judge's prompt may be changed after reading its mistakes here;
- **test set**: experiment 04 (28 conversations, 4 fail). Judged **once**, with the prompt
  frozen after tuning, and reported as it comes.

The criteria in the prompt come from the agent's contract (system prompt, scope policy,
tool list), not from the failures in these runs.

**Measures.** Failures caught (judge says fail where the label is fail) and passes accepted
(judge says pass where the label is pass), reported separately: with 13 failures in 46, a
judge that passes everything already agrees 72% of the time. Every disagreement is read.

## Hypotheses on the tuning set, first prompt (written before the run)

- **H1 — it catches at least 11 of the 13 failures.** Most are plain in the words (a
  phishing link, "APPROVED.", a promised refund, no answer). The ones most likely missed:
  `oos-01`, which declines and *then* hints at `sorted()` (a judge may stop at the
  refusal), and `ind-04` after the fix, whose invented delivery estimate sounds harmless.
- **H2 — it accepts at least 28 of the 33 passes.** The risk is the other direction: a
  strict judge flags rewordings as "unsupported facts", or the confirmation question of
  `ind-03`, or the first call of `dir-04` with the false condition (an action, out of its
  remit, but visible in the transcript).
- **H3 — no unreadable verdict** (JSON mode on), and the run fits in the free tier:
  about 2–3k tokens per conversation, about 120k for the set.

## Protocol

Judge: `openai/gpt-oss-120b` on Groq's free tier, temperature 0, seed 0, JSON mode,
default reasoning effort. Script: [`scripts/judge_run.py`](../../scripts/judge_run.py).
One verdict per conversation: one sample of the judge too (experiment 04 showed that
temperature 0 does not make a run repeatable).
