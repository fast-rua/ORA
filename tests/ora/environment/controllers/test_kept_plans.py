"""The planner a session asks (``controllers/kept_plans.py``): what a dry run planned is what the real call runs."""
from ora.environment.controllers import kept_plans
from ora.environment.controllers.kept_plans import KeptPlans


class Planner:
    """Each solve gives a different answer, as a sampling planner does."""

    def __init__(self):
        self.solved = 0
        self.binding = {"name": "planner"}

    def plan(self, request):
        self.solved += 1
        return {"answer": self.solved}

    solve_ik = plan


REQUEST = {"goal": [1.0], "start": [0.0], "world": {"input": {"path": "/a/world.npz", "sha256": "w1"}}}


def test_the_real_call_gets_what_the_dry_run_got_once():
    kept = KeptPlans(Planner())
    with kept.previewing():
        previewed = kept.plan(REQUEST)
    assert kept.plan(REQUEST) == previewed and kept.reused == 1
    assert kept.plan(REQUEST) != previewed, "only once: a second call of the same is a new plan"


def test_a_plan_is_not_kept_unless_a_dry_run_asked():
    kept = KeptPlans(Planner())
    first = kept.plan(REQUEST)
    assert kept.plan(REQUEST) != first and kept.reused == 0


def test_a_world_is_named_by_its_hash_wherever_its_snapshot_was_written():
    kept = KeptPlans(Planner())
    with kept.previewing():
        previewed = kept.plan(REQUEST)
    moved = {**REQUEST, "world": {"input": {"path": "/b/other-name.npz", "sha256": "w1"}}}
    assert kept.plan(moved) == previewed


def test_any_change_of_the_world_or_the_start_plans_again():
    kept = KeptPlans(Planner())
    with kept.previewing():
        previewed = kept.plan(REQUEST)
    assert kept.plan({**REQUEST, "world": {"input": {"path": "/a/world.npz", "sha256": "w2"}}}) != previewed
    assert kept.plan({**REQUEST, "start": [0.1]}) != previewed
    assert kept.reused == 0


def test_a_receipt_says_when_the_plan_was_kept_and_the_next_one_does_not():
    kept = KeptPlans(Planner())
    with kept.previewing():
        kept.plan(REQUEST)
    kept.plan(REQUEST)
    assert kept_plans.stamped(kept, {})["ran_previewed_plan"] is True
    assert kept_plans.stamped(kept, {}) == {}


def test_the_rest_of_the_planner_is_the_planner():
    assert KeptPlans(Planner()).binding == {"name": "planner"}
    assert kept_plans.stamped(None, {"a": 1}) == {"a": 1}
    with kept_plans.previewing(None):
        pass


def test_old_answers_do_not_pile_up():
    kept = KeptPlans(Planner())
    with kept.previewing():
        for index in range(kept_plans.KEPT_PLANS + 3):
            kept.plan({**REQUEST, "goal": [float(index)]})
    assert len(kept._kept) == kept_plans.KEPT_PLANS
