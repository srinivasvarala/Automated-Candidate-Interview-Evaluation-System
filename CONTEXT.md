# AI Interview Coach

A mock-interview app. A candidate picks a role, answers questions from an AI Interviewer, gets feedback from an AI Evaluator after each answer, and receives a Scorecard at the end.

## Language

**Interview session**:
One run of the app from setup to Scorecard, identified by a session id (also the graph `thread_id`). Has a status: `in_progress`, `completed` or `abandoned`.
_Avoid_: interview (ambiguous with the saved record), conversation, chat

**Interview connection**:
The browser-side module that holds the WebSocket to one Interview session and the saved session id. It sends answers, reconnects with backoff, resumes after a refresh, and turns server events into the state the screens show. Not the same as the server's Interview session.
_Avoid_: client, socket

**Turn**:
One question from the Interviewer, the candidate's answer, and the Evaluator's feedback.
_Avoid_: round, exchange

**Transcript**:
The ordered messages of an Interview session, saved after every Turn.
_Avoid_: history (that means the list of past Interview sessions), log

**Scorecard**:
The structured end-of-session summary: scores per category, strengths, improvements, and the difficulty curve. If the model's output cannot be validated after one retry, the Scorecard is still delivered with no category scores, a plain note, and the difficulty curve.
_Avoid_: summary, report

**Interviewer / Evaluator**:
The two roles that speak. The Interviewer asks; the Evaluator gives feedback and a hidden 1-10 score that drives difficulty.
_Avoid_: agent, bot, coach

**Score**:
The Evaluator's hidden 1-10 rating of one answer. It is not the headline number the candidate sees; it drives the next Tier.
_Avoid_: grade, rating

**Unscored Turn**:
A Turn whose Score could not be obtained from the Evaluator's model, even after one retry. It has feedback but no Score; it is skipped by Difficulty (neither extends nor breaks a Streak) and shown without a score.

**Strong answer / weak answer**:
An answer with a Score of 7 or more / 4 or less.

**Streak**:
Consecutive strong (or weak) answers. A streak of two or more moves the next Tier up (or down) one step, and keeps moving it with each further answer in the streak.

**Tier**:
One of five difficulty levels (Warm-up, Foundation, Mid-level, Advanced, Expert) chosen from the candidate's Scores so far.
_Avoid_: level

**Difficulty**:
The module that turns the Scores so far into the next Tier, the prompt brief for the Interviewer, and the per-question curve on the Scorecard. Not a synonym for Tier.

## Relationships

- An **Interview session** has many **Turns**; its **Transcript** is the messages of all Turns.
- An **Interview session** ends with exactly one **Scorecard** unless it is `abandoned`.
- The **Evaluator's** **Score** for a **Turn** decides the **Tier** of the next **Turn**, through **Difficulty**.
