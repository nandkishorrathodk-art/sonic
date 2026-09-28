from sonic.organization.architecture import (
    DEPARTMENTS,
    SUBSYSTEM_OWNERSHIP,
    Department,
    Team,
    owner_for_path,
    ownership_for_path,
    validate_architecture,
)


def test_architecture_has_seven_departments_and_unique_subsystems() -> None:
    validate_architecture()
    assert len(DEPARTMENTS) == 7
    assert len(SUBSYSTEM_OWNERSHIP) >= 35
    assert len({item.subsystem for item in SUBSYSTEM_OWNERSHIP}) == len(SUBSYSTEM_OWNERSHIP)


def test_core_paths_resolve_to_authoritative_owners() -> None:
    assert owner_for_path("sonic-core/sonic/brain/reasoning.py")[0] == Department.COGNITIVE_SYSTEMS
    assert owner_for_path("sonic-core/sonic/computer_use/boss.py")[0] == Department.COMPUTER_INTERACTION
    assert owner_for_path("sonic-core/sonic/api/main.py")[0] == Department.PLATFORM_SYSTEMS
    assert owner_for_path("sonic-core/sonic/safety/action_policy.py")[0] == Department.SECURITY_SAFETY
    assert owner_for_path("sonic-dashboard/app/page.tsx")[0] == Department.EXPERIENCE_INTERFACE
    assert owner_for_path("sonic-core/sonic/researcher/worker.py")[0] == Department.RESEARCH_EVOLUTION
    assert owner_for_path("sonic-core/sonic/kernel/lib.rs")[1] == Team.SYSTEMS_RUNTIME


def test_unknown_path_fails_closed() -> None:
    try:
        ownership_for_path("random/unowned/path.py")
    except KeyError:
        return
    raise AssertionError("Unknown subsystem must not receive implicit ownership")
