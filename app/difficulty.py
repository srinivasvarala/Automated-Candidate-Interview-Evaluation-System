"""Difficulty module.

Turns the Scores so far into the next Tier, the prompt brief for the
Interviewer, and the per-question curve shown on the Scorecard. The tier and
streaks are never stored: they are derived by replaying the rules over the
list of Scores.
"""

from dataclasses import dataclass

STRONG_AT = 7  # a Score at or above this is a strong answer
WEAK_AT = 4  # a Score at or below this is a weak answer
STREAK = 2  # consecutive strong (or weak) answers that move the Tier

# index 0..4; names match what the UI shows
TIERS = (
    ("Warm-up", "Ask a straightforward question to ease the candidate in and build rapport. Light behavioral or motivational."),
    ("Foundation", "Probe relevant experience or a core concept for the role. Moderate depth."),
    ("Mid-level", "Ask role-specific technical or problem-solving questions that require structured thinking."),
    ("Advanced", "Ask questions requiring multi-step analysis, system-level thinking, or nuanced trade-offs."),
    ("Expert", "Present complex, ambiguous scenarios. Probe edge cases, failure modes, and architectural decisions. This is where strong candidates differentiate themselves."),
)


@dataclass(frozen=True)
class Tier:
    index: int
    name: str
    guidance: str


def _tier(index: int) -> Tier:
    return Tier(index, *TIERS[index])


@dataclass(frozen=True)
class Difficulty:
    scores: tuple[int, ...] = ()

    def __init__(self, scores=()):
        object.__setattr__(self, "scores", tuple(scores))

    def record(self, score: int) -> "Difficulty":
        return Difficulty(self.scores + (score,))

    @property
    def tier(self) -> Tier:
        """The Tier for the next question."""
        return _tier(self._tier_indexes()[-1])

    def curve(self) -> list[dict]:
        """One entry per answered question: its Score and the Tier it was asked at."""
        return [
            {"question": i + 1, "score": score, "tier": TIERS[tier][0], "tier_index": tier}
            for i, (score, tier) in enumerate(zip(self.scores, self._tier_indexes()))
        ]

    def brief(self) -> str:
        """The prompt block telling the Interviewer how hard to ask."""
        tier = self.tier
        text = f"## Difficulty Level: {tier.name}\n{tier.guidance}"
        if not self.scores:
            return text
        avg = sum(self.scores) / len(self.scores)
        if len(self.scores) >= 2 and self.scores[-1] > self.scores[-2]:
            trend = "improving"
        elif len(self.scores) >= 2 and self.scores[-1] < self.scores[-2]:
            trend = "declining"
        else:
            trend = "stable"
        text += f"\n\nThe candidate's performance so far: average score {avg:.1f}/10, trend is {trend}."
        if avg >= STRONG_AT:
            text += "\nThe candidate is performing strongly — challenge them with harder, more nuanced questions."
        elif avg <= WEAK_AT:
            text += "\nThe candidate is struggling — ask a question that lets them demonstrate knowledge without overwhelming complexity."
        return text

    def _tier_indexes(self) -> list[int]:
        """Tier index of each question asked so far, plus the next one (len = scores + 1)."""
        indexes = [0]
        strong = weak = 0
        for score in self.scores:
            strong = strong + 1 if score >= STRONG_AT else 0
            weak = weak + 1 if score <= WEAK_AT else 0
            tier = indexes[-1]
            if strong >= STREAK:
                tier = min(tier + 1, len(TIERS) - 1)
            elif weak >= STREAK:
                tier = max(tier - 1, 0)
            indexes.append(tier)
        return indexes
