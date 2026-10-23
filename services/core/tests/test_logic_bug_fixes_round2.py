"""Regression tests for the second logic-bug audit round.

Covers four real defects found by manual inspection (unit tests passed while
these were live):

  BUG 1 — GUI-burst failure path called ``_inject_replan(goal=<observation>)``,
          so the LLM was told "Goal remains: <error text>" and lost the real
          objective when asked to pivot.
  BUG 2 — ``rm -rf`` on several top-level directories (/run, /snap, /data,
          /app, /workspace, /project, /www) classified as L0_SAFE, so an agent
          could delete a whole mounted volume without a gate.
  BUG 3 — the workstation mission route computed ``is_private_net`` and then
          discarded it, passing ``allow_private_networks=True`` unconditionally,
          widening the egress envelope for every mission.
  BUG 4 — ``POST /experiments/{id}/approve`` ignored the ``update_status``
          result and returned success even when the experiment did not exist.
"""
import inspect

import pytest

from sonic.safety.action_policy import ActionPolicy
from sonic.safety.scope import RiskLevel, ScopeChecker


# ---- BUG 1: replan must carry the objective, never observation text ----
def test_replan_helper_prefers_mission_goal():
    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent.__new__(ComputerUseAgent)
    agent._last_mission_goal = "enumerate the target's web routes"
    assert agent._current_goal_for_replan() == "enumerate the target's web routes"


def test_replan_helper_falls_back_to_goal_attribute():
    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent.__new__(ComputerUseAgent)
    agent._last_mission_goal = None
    agent.goal = "install the requested package"
    assert agent._current_goal_for_replan() == "install the requested package"


def test_replan_helper_never_returns_empty():
    from sonic.computer_use.agent import ComputerUseAgent

    agent = ComputerUseAgent.__new__(ComputerUseAgent)
    agent._last_mission_goal = None
    assert agent._current_goal_for_replan() == "(objective unavailable)"


def test_gui_burst_failure_path_does_not_pass_observation_as_goal():
    from sonic.computer_use.agent import ComputerUseAgent

    source = inspect.getsource(ComputerUseAgent)
    assert "self._inject_replan(goal=str(actual_obs_str))" not in source, (
        "GUI-burst failure must not feed observation text into the replan goal"
    )
    assert "self._inject_replan(goal=self._current_goal_for_replan())" in source


# ---- BUG 2: top-level directory deletion is L2 ----
@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /run",
        "rm -rf /snap",
        "rm -rf /data",
        "rm -rf /app",
        "rm -rf /workspace",
        "rm -rf /project",
        "rm -rf /www",
    ],
)
def test_top_level_directory_deletion_is_forbidden(command):
    assert ScopeChecker().classify_command_risk(command) == RiskLevel.L2_FORBIDDEN


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build/",
        "rm -rf ./node_modules",
        "rm -rf /tmp/scratch",
    ],
)
def test_workspace_internal_deletion_still_allowed(command):
    assert ScopeChecker().classify_command_risk(command) == RiskLevel.L0_SAFE


# ---- BUG 3: private networks only allowed when the engagement is private ----
def _evaluate(policy: ActionPolicy, target: str) -> bool:
    return policy.evaluate("BROWSER_NAVIGATE", target, {}).allowed


def test_private_target_blocked_for_public_engagement():
    policy = ActionPolicy(workspace_root="/tmp", allow_private_networks=False)
    assert _evaluate(policy, "http://10.10.50.10/admin") is False
    assert _evaluate(policy, "http://192.168.1.5/") is False


def test_private_target_allowed_for_private_engagement():
    policy = ActionPolicy(workspace_root="/tmp", allow_private_networks=True)
    assert _evaluate(policy, "http://10.10.50.10/admin") is True


def test_metadata_and_loopback_always_blocked_even_for_private_engagement():
    policy = ActionPolicy(workspace_root="/tmp", allow_private_networks=True)
    assert _evaluate(policy, "http://169.254.169.254/latest/meta-data/") is False
    assert _evaluate(policy, "http://127.0.0.1:8080/") is False


def test_workstation_route_wires_private_net_flag():
    from sonic.api.routes import workstation

    source = inspect.getsource(workstation)
    assert "allow_private_networks=is_private_net" in source, (
        "mission policy must derive private-network allowance from the engaged target"
    )


# ---- BUG 4: experiment lifecycle routes honor the update result ----
@pytest.mark.parametrize(
    "func_name",
    ["rollback_experiment", "approve_experiment", "reject_experiment"],
)
def test_experiment_lifecycle_routes_check_update_status(func_name):
    from sonic.api.routes import experiments

    source = inspect.getsource(getattr(experiments, func_name))
    assert "if not ok" in source, f"{func_name} must fail when update_status returns False"
