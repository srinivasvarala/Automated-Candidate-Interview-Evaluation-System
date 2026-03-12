from app.graph import _after_candidate


class TestAfterCandidate:
    def test_routes_to_summary_when_end_requested(self):
        state = {"end_requested": True, "questions_asked": 1}
        assert _after_candidate(state) == "summary"

    def test_routes_to_evaluator_when_not_ended(self):
        state = {"end_requested": False, "questions_asked": 1}
        assert _after_candidate(state) == "evaluator"

    def test_routes_to_evaluator_when_key_missing(self):
        state = {"questions_asked": 1}
        assert _after_candidate(state) == "evaluator"
