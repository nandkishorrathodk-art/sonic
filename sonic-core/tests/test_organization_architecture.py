from sonic.organization.architecture import (
    DEPARTMENTS,
    SUBSYSTEM_OWNERSHIP,
    Department,
    Team,
    owner_for_path,
    ownership_for_path,
    validate_architecture,
)


def test_architecture_has_seven_departments() -> None:
    assert len(DEPARTMENTS) == 7
    assert set(DEPARTMENTS) == set(Department)


def test_subsystems_have_one_authoritative_owner() -> None:
    validate_architecture()
    names = [item.subsystem for item in SUBSYSTEM_OWNERSHIP]
    assert len(names) == len(set(names))
    assert len(names) >= 35


def test_core_paths_resolve_to_expected_departments() -> None:
    assert owner_for_path("sonic-core/sonic/brain/reasoning.py")[0] == Department.COGNITIVE_SYSTEMS
    assert owner_for_path("sonic-core/sonic/computer_use/boss.py")[0] == Department.COMPUTER_INTERACTION
    assert owner_for_path("sonic-core/sonic/api/main.py")[0] == Department.PLATFORM_SYSTEMS
    assert owner_for_path("sonic-core/sonic/safety/action_policy.py")[0] == Department.SECURITY_SAFETY
    assert owner_for_path("sonic-core/sonic/evidence/custody.py")[0] == Department.EVALUATION_RELIABILITY
    assert owner_for_path("sonic-dashboard/app/page.tsx")[0] == Department.EXPERIENCE_INTERFACE
    assert owner_for_path("sonic-core/sonic/researcher/worker.py")[0] == Department.RESEARCH_EVOLUTION
    assert owner_for_path("sonic-core/sonic/organization/store.py")[0] == Department.PLATFORM_SYSTEMS


def test_team_mapping_is_semantic_not_language_specific() -> None:
    department, team, subsystem = owner_for_path("sonic-core/sonic/kernel/lib.rs")
    assert department == Department.PLATFORM_SYSTEMS
    assert team == Team.SYSTEMS_RUNTIME
    assert subsystem == "kernel"


def test_unknown_path_is_not_silently_assigned() -> None:
    try:
        ownership_for_path("random/unowned/path.py")
    except KeyError:
        return
    raise AssertionError("Unknown subsystem must not receive implicit ownership")
