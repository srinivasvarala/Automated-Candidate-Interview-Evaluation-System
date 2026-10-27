import pytest

from app.difficulty import Difficulty


def play(*scores):
    """Tier index used for each question, for a candidate who scores `scores`."""
    return [entry["tier_index"] for entry in Difficulty(scores).curve()]


def test_starts_at_warm_up():
    assert Difficulty().tier.index == 0
    assert Difficulty().tier.name == "Warm-up"
    assert Difficulty().curve() == []


def test_record_returns_a_new_value():
    start = Difficulty()
    after = start.record(8)
    assert start.scores == ()
    assert after.scores == (8,)


def test_two_strong_answers_move_up_one_tier():
    assert Difficulty([8, 7]).tier.index == 1


def test_two_weak_answers_move_down_one_tier():
    # climb to tier 2 first, then fail twice
    assert Difficulty([9, 9, 9, 3, 4]).tier.index == 1


def test_a_mid_score_breaks_the_streak():
    assert Difficulty([8, 5, 8]).tier.index == 0


def test_stays_within_the_five_tiers():
    assert Difficulty([9] * 12).tier.index == 4
    assert Difficulty([1] * 12).tier.index == 0


def test_streak_keeps_climbing_with_each_further_strong_answer():
    # the streak is not reset on promotion, so every answer after the second climbs
    assert play(9, 9, 9, 9, 9) == [0, 0, 1, 2, 3]
    assert Difficulty([9] * 5).tier.index == 4


def test_streak_keeps_dropping_with_each_further_weak_answer():
    assert play(9, 9, 9, 9, 9, 2, 2, 2) == [0, 0, 1, 2, 3, 4, 4, 3]
    assert Difficulty([9] * 5 + [2, 2, 2]).tier.index == 2


def test_boundaries_are_inclusive():
    assert Difficulty([7, 7]).tier.index == 1  # 7 is strong
    assert Difficulty([6, 6]).tier.index == 0  # 6 is neither
    assert Difficulty([9, 9, 9, 4, 4]).tier.index == 1  # 4 is weak
    assert Difficulty([9, 9, 9, 5, 5]).tier.index == 2  # 5 is neither


def test_curve_pairs_each_score_with_the_tier_it_was_asked_at():
    assert Difficulty([8, 8, 3]).curve() == [
        {"question": 1, "score": 8, "tier": "Warm-up", "tier_index": 0},
        {"question": 2, "score": 8, "tier": "Warm-up", "tier_index": 0},
        {"question": 3, "score": 3, "tier": "Foundation", "tier_index": 1},
    ]


def test_brief_names_the_tier_with_no_scores_yet():
    brief = Difficulty().brief()
    assert "Warm-up" in brief
    assert "performance so far" not in brief


@pytest.mark.parametrize(
    "scores, expected, absent",
    [
        ([9, 8], "challenge them", "struggling"),
        ([2, 3], "struggling", "challenge them"),
        ([5, 6], "average score 5.5/10", "challenge them"),
    ],
)
def test_brief_reports_performance(scores, expected, absent):
    brief = Difficulty(scores).brief()
    assert expected in brief
    assert absent not in brief


def test_brief_reports_the_trend():
    assert "trend is improving" in Difficulty([5, 7]).brief()
    assert "trend is declining" in Difficulty([7, 5]).brief()
    assert "trend is stable" in Difficulty([6, 6]).brief()
    assert "trend is stable" in Difficulty([6]).brief()


def test_an_unscored_turn_neither_extends_nor_breaks_a_streak():
    assert Difficulty([8, None, 8]).tier.index == 1
    assert Difficulty([8, None]).tier.index == 0


def test_an_unscored_turn_keeps_the_tier_and_shows_in_the_curve():
    assert Difficulty([9, 9, None]).curve()[-1] == {
        "question": 3, "score": None, "tier": "Foundation", "tier_index": 1,
    }


def test_brief_ignores_unscored_turns():
    assert "average score 8.0/10" in Difficulty([8, None]).brief()
    assert "performance so far" not in Difficulty([None]).brief()
