from avcs_suite_a.catalog import CASES, CASE_BY_ID, validate_catalog


def test_catalog_has_unique_valid_ids_and_targets():
    assert not validate_catalog()
    assert len(CASES) == len(CASE_BY_ID)
    assert len(CASES) >= 12


def test_negative_control_is_marked_blocked():
    case = CASE_BY_ID["disabled_control"]
    assert case.expected_status == "blocked"
    assert case.expected_state == {"continued": False}
    assert not next(c for c in case.controls if c.id == "continue").enabled


def test_catalog_does_not_duplicate_expected_target_ids():
    for case in CASES:
        if case.target_id is not None:
            assert case.target_id in {control.id for control in case.controls}
