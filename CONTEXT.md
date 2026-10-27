# AI Interview Coach

A mock-interview app. A candidate picks a role, answers questions from an AI Interviewer, gets feedback from an AI Evaluator after each answer, and receives a Scorecard at the end.

## Language

**Interview session**:
One run of the app from setup to Scorecard, identified by a session id (also the graph `thread_id`). Has a status: `in_progress`, `completed` or `abandoned`.
_Avoid_: interview (ambiguous with the saved record), conversation, chat

**Turn**:
One question from the Interviewer, the candidate's answer, and the Evaluator's feedback.
_Avoid_: round, exchange

**Transcript**:
The ordered messages of an Interview session, saved after every Turn.
_Avoid_: history (that means the list of past Interview sessions), log

**Scorecard**:
The structured end-of-session summary: scores per category, strengths, improvements, and the difficulty curve.
_Avoid_: summary, report

**Interviewer / Evaluator**:
The two roles that speak. The Interviewer asks; the Evaluator gives feedback and a hidden 1-10 score that drives difficulty.
_Avoid_: agent, bot, coach

**Tier**:
One of five difficulty levels (Warm-up, Foundation, Mid-level, Advanced, Expert) chosen from the candidate's recent scores.
_Avoid_: level, difficulty (as a noun for the tier itself)

## Relationships

- An **Interview session** has many **Turns**; its **Transcript** is the messages of all Turns.
- An **Interview session** ends with exactly one **Scorecard** unless it is `abandoned`.
- The **Evaluator's** score for a **Turn** decides the **Tier** of the next **Turn**.
