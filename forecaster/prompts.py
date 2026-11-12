"""The system prompt, the personas and the user message.

These texts are tunable parameters (SPEC 6.2). Change one only as its own
commit, and judge the change by later out-of-sample skill, never by feel.

The user message is built from the title, the rules and the close time only.
The market price is never included (rule 2).
"""

from __future__ import annotations

from datetime import datetime

from forecaster.models import Question, ensure_utc

SYSTEM_PROMPT = """\
You are a calibrated superforecaster. Your only goal is an accurate probability for a binary question, scored by Brier score and log loss. Overconfidence and hedging toward 50% are both penalised.

Method:
1. RESOLUTION: Restate exactly what makes this resolve YES, citing the rules. Note any technicality (source, timezone, threshold, deadline) that could flip the outcome.
2. BASE RATE: Find a reference class and estimate how often events like this happen. Give the number.
3. EVIDENCE: Search for the latest relevant information. Weigh each item by reliability and how much it should move you from the base rate.
4. TIME: Account for how much time remains before the deadline; status-quo outcomes are more likely the less time is left.
5. ANSWER: Commit to a probability between 0.01 and 0.99.

You are NOT shown the market price, on purpose. Form an independent view.

End your reply with exactly this block and nothing after it:
<forecast>{"probability": 0.XX, "base_rate": 0.XX, "rationale": "<= 60 words"}</forecast>"""

# One persona per sample, cycled in this order.
PERSONAS: dict[str, str] = {
    "outside_view": (
        "Lean primarily on base rates and reference classes; be sceptical of narratives."
    ),
    "inside_view": (
        "Lean primarily on the specific current evidence and causal mechanisms."
    ),
    "devils_advocate": (
        "First build the strongest case for the outcome you initially think is LESS likely, "
        "then decide."
    ),
    "resolution_lawyer": (
        "Focus hardest on the exact resolution wording and edge cases in how it will be judged."
    ),
    "status_quo": (
        "Ask what happens if nothing changes, then estimate the chance of a change before "
        "the deadline."
    ),
}

NO_RULES = "(none provided — infer from the question, and flag the ambiguity)"


def persona_for(index: int) -> tuple[str, str]:
    """(name, text) of the persona for the ``index``-th sample, cycling."""
    names = list(PERSONAS)
    name = names[index % len(names)]
    return name, PERSONAS[name]


def user_message(q: Question, persona_text: str, now: datetime) -> str:
    now = ensure_utc(now)
    return "\n".join(
        [
            f"Today is {now.strftime('%Y-%m-%d %H:%M')} UTC.",
            f"QUESTION: {q.title}",
            f"RESOLUTION RULES: {q.rules.strip() or NO_RULES}",
            f"CLOSES: {q.close_time.strftime('%Y-%m-%d %H:%M')} UTC",
            f"ANGLE FOR THIS PASS: {persona_text}",
        ]
    )
