# Experiment 04 — the reference tasks, first run

**Question.** On what customers normally ask (order status, returns, refunds, policy
questions), how often does the agent end in the right **database state**, and how often
is its answer right when read by hand? This is the first run of the evaluation set
([`evaluation/tasks.json`](../../evaluation/tasks.json): 24 tasks, plus the 4 cases found
in chat sessions). It also tells the judge (step 6b) what it will have to catch.

## Hypotheses (written before the run)

- **H1 — refunds that should happen, happen.** Of the 5 tasks where the reviewer approves,
  at least 4 end with exactly the expected refund row. The risk: the model may stop to ask
  "would you like me to process this refund?" instead of calling `refund_order`, even when
  the customer wrote "please refund it" (seen in experiment 03, `ind-03`, after "can I get
  a refund?"). A task that ends on a question writes nothing and fails.
- **H2 — the condition decides the amount.** In the 3 defective-item tasks the model uses
  `defective_or_wrong`, so the shipping fee is included where there is one (`O10041`,
  `O10090`); choosing `changed_mind` would write 45.00 instead of 50.90.
- **H3 — the final-state check is weak on most tasks.** 23 of 28 tasks expect no refund.
  Any request beyond the reviewer's script is rejected, so those tasks pass the state
  check whatever the agent says. Their real result is in the answers.
- **H4 — answers: at least 20 of 28 right by hand.** Expected failures: `chat-02` (the
  rule on defective items is cut from the policy search), `chat-03` (promises the outcome
  after a rejection), an invented delivery date on `t-status-shipped` (seen in experiment
  03), and possibly `t-cancel-processing-no-tool` claiming the order is cancelled.
- **H5 — at most one turn without an answer** (the search loop of experiment 03, `leg-03`).

## Protocol

Same model, server and settings as experiments 01–03: Qwen3.5-4B Q4_K_M, llama-server
b12100, `-c 8192 -t 4`, thinking off, temperature 0. Laptop on AC power. Code: the commit
that adds this file. Runner: [`scripts/run_eval.py`](../../scripts/run_eval.py)
`--set tasks`, one conversation per task on a private copy of the database, scripted
reviewer. Every answer is read by hand against its `expected` field. One run per task:
one sample, not a rate.

## Results

File: [`traces.jsonl`](traces.jsonl). One run per task, temperature 0.

**Final-state check: 26/28. Read by hand: 22/28.**

| Category | State check | By hand | Failures by hand |
|---|---|---|---|
| Order status | 8/8 | 8/8 | — |
| Returns | 5/5 | 5/5 | — |
| Refunds | 6/8 | 5/8 | final-sale-defective, over-two-turns (no refund); rejected-by-reviewer (promise) |
| Policy | 3/3 | 1/3 | refund-method, cancel-processing |
| From chat sessions | 4/4 | 3/4 | chat-03 (promise) |

- **H1 refuted: 3 of 5 approved refunds were written.** In the two others
  (`t-refund-final-sale-defective`: "...arrived with a broken seam. Refund please.";
  `t-refund-over-two-turns`: "...arrived chipped. I'd like a refund.") the model checked
  eligibility, got the right quote, then stopped: "Would you like me to process this
  refund now?". The customer had asked; the human approval is the confirmation the policy
  requires (refunds rule 4). Nothing harmful, but the refund needs one more turn, about a
  minute on this CPU. Where the customer wrote "Please refund it / one", it went ahead
  every time (3/3); two samples cannot say whether the wording is the cause.
- **H2 supported.** All three defective-item tasks used `defective_or_wrong`, and the
  quotes included the shipping fee where there was one (50.90, 39.90).
- **H3 supported: 4 tasks pass the state check and fail by hand**, all in the answer:
  - `t-refund-rejected-by-reviewer` and `chat-03`: "Once that's submitted, we can process
    the refund": the same promise after a rejection as in the chat session (the agent
    cannot receive a photo and does not decide);
  - `t-policy-refund-method`: called no tool and answered "I do not have information about
    the specific refund method" although refunds rule 1 answers it;
  - `t-cancel-processing-no-tool`: "Would you like me to proceed with the cancellation
    request?": offers an action it has no tool for.
- **H4 supported (22 ≥ 20), but the predicted failures were partly wrong.** `chat-02`
  now passes: the model searched "return", then "defective", and stated the rule it had
  missed in the chat. `t-status-shipped` invented no delivery date. `chat-01` answered the
  push-back with "Yes, I am sure", without the apology seen in experiment 02.
- **H5 supported**: every turn got an answer.

Also seen: "145.00 EUR will be credited to your account" (`t-refund-rest-of-partly-refunded`)
is vague where the policy is precise (original payment method); counted as right.

**Timings** (on AC power): 7–124 s per task, median about 50 s; refunds and multi-turn
tasks are the slowest (4–6 model calls).

## Conclusion

On what customers normally ask, the agent gets the facts right: statuses, return windows
(day 30 and day 31 included), refused refunds, amounts. Its failures are of two kinds:

1. **It stops before acting** (2 of 5 refunds): a confirmation question where the request
   was explicit. Visible in the database state.
2. **Its words go beyond what it can do** (4 tasks): promises after a rejection, an offer
   to cancel without a tool, "I don't know" without looking. Invisible in the database
   state: they are what the judge (step 6b) has to catch.

The final-state check alone gives 26/28; reading the answers gives 22/28. On this set the
database is the right check for refunds and says almost nothing about the rest.
