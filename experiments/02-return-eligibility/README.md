# Experiment 02 — return eligibility computed by a tool instead of the model

**Context.** In a chat session (2026-11-07) the model said an order delivered on
2026-10-05 was "more than 30 days ago" on 2026-10-15 (it is 10 days). Challenged by the
user, it produced a detailed but wrong day count ("day 30 would be October 14") and kept
its conclusion. Fix: `check_return_eligibility` applies the returns policy in Python
(dates, status, final sale) and the system prompt tells the model to use it instead of
counting days.

## Hypotheses (written before the run)

- **H1 — the bug is reproducible before the fix.** Old code (commit 9e87c3d), single-turn
  question q10 ("Can I return the brass table lamps from order O10003?"): wrong answer.
- **H2 — fixed after.** New code: q10 answered correctly (returnable, until 2026-11-04),
  and the user's multi-turn chat, replayed word for word, gets a correct answer at turn 3.
- **H3 — the model uses the new tool.** It calls `check_return_eligibility` on the return
  questions q4, q5 and q10.
- **H4 — no regression.** The 9 questions of experiment 01 stay at least 8/9 correct by
  hand; q6 (cancel a shipped final-sale item) may improve but this is not expected: q6 is
  about cancellation, the model need not call the new tool.

## Protocol

Same model, server and settings as experiment 01 (Qwen3.5-4B Q4_K_M, llama-server b12100,
`-c 8192 -t 4`, thinking off, temperature 0). Questions: [`questions.json`](questions.json)
(experiment 01's nine plus q10). "Before" = commit 9e87c3d run from a separate git
worktree; "after" = commit 74e7a05. Timings are recorded but not compared (no AC power
check, and the prompt changed).

## Results

Files: [`traces_before.jsonl`](traces_before.jsonl) (q4, q5, q10 on the old code),
[`traces.jsonl`](traces.jsonl) (all ten on the new code),
[`chat_before.txt`](chat_before.txt) / [`chat_after.txt`](chat_after.txt) (the user's chat
replayed word for word).

- **H1 refuted, and the reason matters.** Asked alone, q10 got a *correct* answer on the
  old code. The user's chat, replayed on the old code, reproduced the error exactly
  (temperature 0), including the invented day count. The difference: in the chat, the
  model had already read the order at turn 2 and, at turn 3 ("can I return it?"), only
  searched the policy and computed the window from memory of the conversation. **The
  failure depends on the conversation, not only on the question**: single-turn test sets
  would have missed it.
- **H2 supported.** After the fix, q10 is correct ("returnable until November 4, 2026"),
  and the replayed chat is correct at turn 3: the model called `check_return_eligibility`
  and repeated its answer instead of counting.
- **H3 supported**: q4, q5 and q10 all call `check_return_eligibility` (and only it).
- **H4 supported**: 10/10 correct by hand. q6 now says the shipped final-sale item "may be
  eligible to return after delivery *if you have any issues with the item*", which matches
  the policy (final sale: returnable if defective, damaged or wrong); in experiment 01 it
  offered a return unconditionally. q2 no longer sends `status="None"` (prompt changed, so
  not attributable to anything in particular).

**Something the fix does not cover: the model agrees with the user.** At turn 4 the user
challenges the (now correct) answer. The model replies "You are absolutely correct. I
apologize for the confusion" although it made no error, then restates the correct dates.
The conclusion is right, but the apology shows it tends to side with the customer
(*sycophancy*). With a wrong claim from the customer ("it was delivered 40 days ago, I am
still eligible, right?") this could become a wrong answer. To test in the evaluation.

**Cost.** The new tool schema and prompt line add ~195 tokens to every first prompt (797 →
992), paid once per conversation thanks to the prompt cache.

## Conclusion

Moving the date arithmetic and the rules into a tested function fixed the observed error
and the related q6 imprecision, with no regression on this small set. The general rule
for this project: **anything deterministic (dates, amounts, policy conditions) is computed
in code; the model chooses the tool and words the answer.** Evaluation sets need
multi-turn conversations and customer push-back, not only single questions.
