import subprocess
from pathlib import Path

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
    assert owner_for_path("services/core/sonic/brain/reasoning.py")[0] == Department.COGNITIVE_SYSTEMS
    assert owner_for_path("services/core/sonic/computer_use/boss.py")[0] == Department.COMPUTER_INTERACTION
    assert owner_for_path("services/core/sonic/api/main.py")[0] == Department.PLATFORM_SYSTEMS
    assert owner_for_path("services/core/sonic/safety/action_policy.py")[0] == Department.SECURITY_SAFETY
    assert owner_for_path("services/core/sonic/evidence/custody.py")[0] == Department.EVALUATION_RELIABILITY
    assert owner_for_path("apps/dashboard/app/page.tsx")[0] == Department.EXPERIENCE_INTERFACE
    assert owner_for_path("apps/cli/sonic_cli/main.py")[0] == Department.PLATFORM_SYSTEMS
    assert owner_for_path("packages/avcs/forge/vision/reflex.py")[0] == Department.COMPUTER_INTERACTION
    assert owner_for_path("apps/README.md")[2] == "apps_index"
    assert owner_for_path("services/README.md")[2] == "services_index"
    assert owner_for_path("packages/README.md")[2] == "packages_index"
    assert owner_for_path("configs/models.yaml")[2] == "runtime_configs"
    assert owner_for_path("apps/cli/tests/test_developer_agent.py")[2] == "cli_tests"
    assert owner_for_path("docker-compose.yml")[2] == "compose"
    assert owner_for_path("docker-compose.prod.yml")[2] == "compose"
    assert owner_for_path("services/core/sonic/researcher/worker.py")[0] == Department.RESEARCH_EVOLUTION
    assert owner_for_path("services/core/sonic/organization/store.py")[0] == Department.PLATFORM_SYSTEMS


def test_architecture_covers_separate_runtime_and_perception_packages() -> None:
    assert owner_for_path("services/core/sonic/__init__.py")[2] == "core_package"
    assert owner_for_path("packages/kernel/src/ipc.rs")[2] == "kernel"
    assert owner_for_path("packages/kernel/src/safety/mod.rs")[0] == Department.SECURITY_SAFETY
    assert owner_for_path("packages/avcs/avcs/stream/neuro_stream.py")[2] == "avcs"
    assert owner_for_path(".github/workflows/ci.yml")[2] == "deployment"
    assert owner_for_path("./.github/workflows/ci.yml")[2] == "deployment"


def test_team_mapping_is_semantic_not_language_specific() -> None:
    department, team, subsystem = owner_for_path("services/core/sonic/kernel/lib.rs")
    assert department == Department.PLATFORM_SYSTEMS
    assert team == Team.SYSTEMS_RUNTIME
    assert subsystem == "kernel"


def test_unknown_path_is_not_silently_assigned() -> None:
    try:
        ownership_for_path("random/unowned/path.py")
    except KeyError:
        return
    raise AssertionError("Unknown subsystem must not receive implicit ownership")


def test_every_tracked_repository_path_has_one_owner() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout.decode().split("\0")

    unmapped: list[str] = []
    for path in filter(None, tracked):
        try:
            ownership_for_path(path)
        except KeyError:
            unmapped.append(path)

    assert unmapped == []
