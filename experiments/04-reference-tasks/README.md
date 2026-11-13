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

(to be written after the run)
