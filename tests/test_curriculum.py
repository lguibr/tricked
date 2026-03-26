from tricked.training.curriculum import evaluate_curriculum


def test_curriculum_steady_state():
    diff, drops, act = evaluate_curriculum(3, 150.0, 140.0, 0)
    assert diff == 3
    assert drops == 0
    assert act == "none"


def test_curriculum_promotion():
    diff, drops, act = evaluate_curriculum(3, 310.0, 305.0, 0)
    assert diff == 4
    assert drops == 0
    assert act == "promote"


def test_curriculum_promotion_cap():
    diff, drops, act = evaluate_curriculum(6, 400.0, 350.0, 0)
    assert diff == 6  # Should not exceed max difficulty 6
    assert act == "none"


def test_curriculum_demotion_tracking():
    # First drop
    diff, drops, act = evaluate_curriculum(4, 45.0, 48.0, 0)
    assert diff == 4
    assert drops == 1
    assert act == "none"

    # Second drop
    diff, drops, act = evaluate_curriculum(4, 45.0, 48.0, 1)
    assert diff == 4
    assert drops == 2
    assert act == "none"

    # Third drop -> Demote!
    diff, drops, act = evaluate_curriculum(4, 45.0, 48.0, 2)
    assert diff == 3
    assert drops == 0
    assert act == "demote"


def test_curriculum_demotion_floor():
    # Attempt to demote below 1
    diff, drops, act = evaluate_curriculum(1, 10.0, 12.0, 2)
    assert diff == 1
    assert drops == 0  # because `else` resets it
    assert act == "none"


def test_curriculum_recovery():
    # Suppose we had 2 drops, but then score improves
    diff, drops, act = evaluate_curriculum(4, 80.0, 75.0, 2)
    assert diff == 4
    assert drops == 0  # Consecutive drops reset
    assert act == "none"
