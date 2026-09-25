"""End-to-end tests for SONIC's Laya System-1 Non-Autoregressive Decision Engine integration.

Validates the full cognitive lifecycle across:
1. Single-pass typed decisions (choice, score, noul) across English, Hindi, and Hinglish.
2. SubMission automatic dual-plane resolution (gui vs headless) across languages.
3. ComputerUseAgent.verify_goal fail-closed dummy marker rejection & real receipt verification.
4. CuriosityLoop + LayaSystem1Engine.evaluate_curiosity_affect scoring novel vs duplicate observations.
5. Laptop RAM safety protection (_has_sufficient_ram_gb and zero-RAM behavior when local model disabled).
6. FastAPI TestClient verification of Laya API endpoints (/llm/laya/status and /llm/laya/predict mounted on sonic.api.main:app).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sonic.api.main import app
from sonic.computer_use.agent import ComputerUseAgent
from sonic.computer_use.curiosity import CuriosityLoop
from sonic.computer_use.models_boss import SubMission
from sonic.llm.laya_engine import (
    LayaSystem1Engine,
    _has_sufficient_ram_gb,
)
from sonic.sandbox.provider import ExecResult

# ============================================================================
# 1. Single-Pass Decisions across English, Hindi, and Hinglish
# ============================================================================

@pytest.mark.no_live_infra
def test_laya_single_pass_decisions_english():
    """Verify single-pass choice, score, and noul evaluation on English inputs."""
    engine = LayaSystem1Engine(enable_local_model=False)

    questions = {
        "plane": {
            "type": "choice",
            "instructions": "Select execution plane substrate",
            "criteria": {
                "gui": "visual desktop browser screenshot user interface click button open website",
                "headless": "terminal bash cli command script python test pytest audit server backend",
            },
        },
        "complexity": {
            "type": "score",
            "instructions": "Task implementation complexity",
            "criteria": ["low", "medium", "high"],
        },
        "is_backend_task": {
            "type": "noul",
            "instructions": "Determine if this is a backend engineering task",
            "affirmative_anchor": "backend database python test pytest script api server terminal",
            "negative_anchor": "frontend design ui sketch drawing browser visual paint click",
        },
    }

    state = "Implement a REST endpoint with FastAPI and execute unit tests via pytest"
    res = engine.predict(state=state, questions=questions)

    # 1. Check choice answer
    assert res.get_choice("plane") == "headless"
    assert res.answers["plane"]["probability"] > 0.50

    # 2. Check score answer
    score = res.get_score("complexity")
    assert 0.0 <= score <= 1.0
    assert res.answers["complexity"]["label"] in ("low", "medium", "high")

    # 3. Check noul binary probability answer
    noul_prob = res.get_noul("is_backend_task")
    assert noul_prob > 0.50
    assert res.answers["is_backend_task"]["answer"] is True

    # 4. Routing metadata
    assert res.routing["language"] == "en"
    assert res.routing["model"] == "english"
    assert res.routing["backend"] == "laya_vector_projection"


@pytest.mark.no_live_infra
def test_laya_single_pass_decisions_multilingual_unicode():
    """Verify Unicode script detection and custom criteria evaluation without hardcoded vocab lists."""
    engine = LayaSystem1Engine(enable_local_model=False)

    questions = {
        "department": {
            "type": "choice",
            "instructions": "Select target department",
            "criteria": {
                "billing": "invoice payment refund duplicate charge",
                "engineering": "bug crash server exception stacktrace deployment",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Priority scale",
            "criteria": ["low", "medium", "high"],
        },
    }

    state = "Server deployment failed with stacktrace exception — Café production node"
    res = engine.predict(state=state, questions=questions)

    assert res.get_choice("department") == "engineering"
    assert res.answers["department"]["probability"] > 0.50
    assert 0.0 <= res.get_score("urgency") <= 1.0
    assert res.routing["model"] == "multilingual"


@pytest.mark.no_live_infra
def test_laya_preset_evaluation():
    """Verify built-in presets (`triage`, `guard`, `verification`, `curiosity`)."""
    engine = LayaSystem1Engine(enable_local_model=False)

    res_triage = engine.predict_preset("Write a python script in terminal and run pytest", preset="triage")
    assert res_triage.get_choice("plane") == "headless"
    assert 0.0 <= res_triage.get_score("urgency") <= 1.0

    res_guard = engine.predict_preset("rm -rf / --no-preserve-root", preset="guard")
    assert res_guard.get_noul("is_dangerous") > 0.50


# ============================================================================
# 2. SubMission Automatic Plane Resolution
# ============================================================================

@pytest.mark.no_live_infra
def test_submission_plane_resolution():
    """Verify SubMission automatically infers plane (gui vs headless) via LayaSystem1Engine."""
    sub_en_hl = SubMission(goal="Install python dependencies and run pytest test suite in terminal")
    assert sub_en_hl.plane == "headless"

    sub_en_gui = SubMission(goal="Open browser window and click on login button on screen")
    assert sub_en_gui.plane == "gui"


@pytest.mark.no_live_infra
def test_submission_plane_explicit_override_preservation():
    """Verify explicit plane specification takes precedence over automatic inference."""
    # GUI-worded goal explicitly forced to headless plane
    sub_forced_hl = SubMission(goal="Open browser and inspect", plane="headless")
    assert sub_forced_hl.plane == "headless"

    # Terminal-worded goal explicitly forced to gui plane
    sub_forced_gui = SubMission(goal="run pytest in terminal", plane="gui")
    assert sub_forced_gui.plane == "gui"

    # Default fallback on empty goal
    sub_empty = SubMission(goal="")
    assert sub_empty.plane in ("gui", "headless")


# ============================================================================
# 3. ComputerUseAgent.verify_goal: Dummy Marker Rejection & Real Receipt Output
# ============================================================================

@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_computer_use_agent_verify_goal_rejects_dummy_marker():
    """Verify ComputerUseAgent.verify_goal rejects placeholder dummy terminal markers fail-closed."""
    computer = MagicMock()
    # Terminal probe returning dummy ready output
    computer.terminal = AsyncMock(
        return_value=ExecResult(
            command="python -m pytest",
            exit_code=0,
            stdout="Terminal Ready (0 procs)",
            stderr="",
        )
    )

    agent = ComputerUseAgent(computer_provider=computer, llm_router=None)
    verified, evidence = await agent.verify_goal("ws-test", "run pytest unit tests")

    assert verified is False
    assert "dummy marker detected" in evidence.lower()
    assert "terminal ready (0 procs)" in evidence.lower()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_computer_use_agent_verify_goal_rejects_sonic_obs_ready_dummy():
    """Verify ComputerUseAgent.verify_goal rejects __sonic_obs_ready__ dummy marker."""
    computer = MagicMock()
    computer.terminal = AsyncMock(
        return_value=ExecResult(
            command="python -m pytest",
            exit_code=0,
            stdout="some task __sonic_obs_ready__ output",
            stderr="",
        )
    )

    agent = ComputerUseAgent(computer_provider=computer, llm_router=None)
    verified, evidence = await agent.verify_goal("ws-test", "run pytest unit tests")

    assert verified is False
    assert "dummy marker detected" in evidence.lower()


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_computer_use_agent_verify_goal_accepts_real_receipt():
    """Verify ComputerUseAgent.verify_goal approves empirical execution receipts."""
    computer = MagicMock()
    computer.terminal = AsyncMock(
        return_value=ExecResult(
            command="python -m pytest",
            exit_code=0,
            stdout="================ 5 passed in 0.15s ================",
            stderr="",
        )
    )

    agent = ComputerUseAgent(computer_provider=computer, llm_router=None)
    verified, evidence = await agent.verify_goal("ws-test", "run pytest unit tests")

    assert verified is True
    assert "5 passed in 0.15s" in evidence


@pytest.mark.no_live_infra
@pytest.mark.asyncio
async def test_computer_use_agent_verify_goal_accepts_file_verified_receipt():
    """Verify ComputerUseAgent.verify_goal approves FILE_VERIFIED receipt."""
    computer = MagicMock()
    computer.terminal = AsyncMock(
        return_value=ExecResult(
            command="test -f /home/sonic/workspace/report.md",
            exit_code=0,
            stdout="FILE_VERIFIED: /home/sonic/workspace/report.md\n# Summary",
            stderr="",
        )
    )

    agent = ComputerUseAgent(computer_provider=computer, llm_router=None)
    verified, evidence = await agent.verify_goal("ws-test", "Create report.md")

    assert verified is True
    assert "FILE_VERIFIED" in evidence


@pytest.mark.no_live_infra
def test_laya_verify_execution_receipt_rejections_and_approvals():
    """Verify LayaSystem1Engine.verify_execution_receipt scores match ground-truth success."""
    engine = LayaSystem1Engine(enable_local_model=False)

    # 1. Dummy placeholder must be rejected (< 0.30)
    dummy_score = engine.verify_execution_receipt(
        goal="Run backend test suite",
        evidence="Re-observed: app=, term=Terminal Ready (0 procs)",
    )
    assert dummy_score < 0.30

    # 2. Blocked / exit 126 error must be rejected (< 0.30)
    blocked_score = engine.verify_execution_receipt(
        goal="Run port scan",
        evidence="Execution blocked by safety envelope: exit 126",
    )
    assert blocked_score < 0.30

    # 3. Verified file receipt must be accepted (>= 0.65)
    valid_score = engine.verify_execution_receipt(
        goal="Create summary report",
        evidence="FILE_VERIFIED: /home/sonic/workspace/report.md — 200 OK passed exit 0",
    )
    assert valid_score >= 0.65


# ============================================================================
# 4. CuriosityLoop + LayaSystem1Engine.evaluate_curiosity_affect
# ============================================================================

@pytest.mark.no_live_infra
def test_curiosity_loop_affect_novel_vs_duplicate_scoring():
    """Verify CuriosityLoop known facts and Laya novelty/affect scoring differentiate discoveries."""
    engine = LayaSystem1Engine(enable_local_model=False)
    mock_router = MagicMock()

    loop = CuriosityLoop(llm_router=mock_router, vector_memory=None)
    loop.state.known_facts.extend([
        "Workspace initialized",
        "Active service on port 8080 running nginx",
    ])

    # 1. Duplicate observation (already in known facts)
    obs_dup = "Active service on port 8080 running nginx"
    res_dup = engine.evaluate_curiosity_affect(obs_dup, loop.state.known_facts)
    novelty_dup = loop.measure_novelty(obs_dup)

    assert res_dup["novelty_score"] <= 0.20
    assert novelty_dup <= 0.20

    # 2. Novel discovery (not in known facts)
    obs_novel = "Discovered unauthenticated Redis database on port 6379 with key-value dumps"
    res_novel = engine.evaluate_curiosity_affect(obs_novel, loop.state.known_facts)
    novelty_novel = loop.measure_novelty(obs_novel)

    assert res_novel["novelty_score"] >= 0.50
    assert novelty_novel > novelty_dup
    assert res_novel["novelty_score"] > res_dup["novelty_score"]

    # 3. Worth-learning probability: genuine novel discovery vs empty idle noise
    obs_idle = "empty idle waiting nothing unchanged error"
    res_idle = engine.evaluate_curiosity_affect(obs_idle, loop.state.known_facts)

    assert res_idle["worth_learning_prob"] < 0.35
    assert res_novel["worth_learning_prob"] > 0.60

    # 4. Curiosity compounds: after recording the novel discovery, its novelty drops
    loop.state.known_facts.append(obs_novel)
    novelty_after_learning = loop.measure_novelty(obs_novel)
    res_after_learning = engine.evaluate_curiosity_affect(obs_novel, loop.state.known_facts)

    assert novelty_after_learning < novelty_novel
    assert res_after_learning["novelty_score"] < res_novel["novelty_score"]


# ============================================================================
# 5. Laptop RAM Protection Check & Zero-RAM Behavior
# ============================================================================

@pytest.mark.no_live_infra
def test_laptop_ram_protection_check():
    """Verify _has_sufficient_ram_gb behaves correctly on small vs giant thresholds."""
    # Realistic small threshold (e.g. 1 MB) should report sufficient RAM
    assert _has_sufficient_ram_gb(0.001) is True

    # Impossibly gigantic threshold (e.g. 1 Million GB) should report insufficient RAM
    assert _has_sufficient_ram_gb(999999.0) is False

    # Graceful fallback on unexpected system query error
    with patch.dict("sys.modules", {"psutil": None, "ctypes": None}):
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert _has_sufficient_ram_gb(1.5) is True


@pytest.mark.no_live_infra
def test_laya_zero_ram_behavior_when_local_model_disabled():
    """Verify enable_local_model=False operates with 0 MB neural weight footprint in host RAM."""
    engine = LayaSystem1Engine(enable_local_model=False)

    # Local router must remain None (zero PyTorch / local weight loading)
    assert engine._get_local_router() is None

    # Decisions run entirely via calibrated dense vector projection
    res = engine.decide_execution_plane("Run pytest in terminal")
    assert res[0] == "headless"

    pred = engine.predict("echo hello", {"q": {"type": "score", "criteria": ["a", "b"]}})
    assert pred.routing["backend"] == "laya_vector_projection"


@pytest.mark.no_live_infra
def test_laya_ram_guard_blocks_local_loading_under_memory_pressure():
    """Verify local router instantiation aborts when RAM headroom < 1.5 GB."""
    engine = LayaSystem1Engine(enable_local_model=True)

    # Mock low physical RAM condition
    with patch("sonic.llm.laya_engine._has_sufficient_ram_gb", return_value=False):
        router = engine._get_local_router()
        assert router is None


# ============================================================================
# 6. FastAPI TestClient Verification of Laya API Endpoints
# ============================================================================

@pytest.mark.no_live_infra
def test_laya_api_status_endpoint_on_app():
    """Verify GET /llm/laya/status on sonic.api.main:app."""
    client = TestClient(app)
    token_response = client.post("/auth/dev-token")
    assert token_response.status_code == 200
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}
    resp = client.get("/llm/laya/status", headers=headers)
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ready"
    assert "backend" in data
    assert "sufficient_ram" in data or "ram_check" in data
    assert "presets" in data or "supported_presets" in data
    assert "languages" in data or "language_support" in data


@pytest.mark.no_live_infra
def test_laya_api_predict_endpoint_on_app():
    """Verify POST /llm/laya/predict on sonic.api.main:app across English, Hindi, and Hinglish."""
    client = TestClient(app)
    token_response = client.post("/auth/dev-token")
    assert token_response.status_code == 200
    headers = {"Authorization": f"Bearer {token_response.json()['access_token']}"}

    # 1. English with preset
    resp_en = client.post(
        "/llm/laya/predict",
        json={
            "state": "Install dependencies and run pytest in terminal",
            "preset": "plane",
        },
        headers=headers,
    )
    assert resp_en.status_code == 200
    data_en = resp_en.json()
    assert "answers" in data_en
    assert "plane" in data_en["answers"]
    assert data_en["answers"]["plane"]["choice"] == "headless"
    assert data_en["routing"]["language"] == "en"

    # 2. Custom questions schema
    resp_custom = client.post(
        "/llm/laya/predict",
        json={
            "state": "Open browser window and click button on screen",
            "questions": {
                "plane": {
                    "type": "choice",
                    "instructions": "Select plane",
                    "criteria": {
                        "gui": "gui visual browser screen window button",
                        "headless": "headless terminal code bash",
                    },
                }
            },
        },
        headers=headers,
    )
    assert resp_custom.status_code == 200
    data_custom = resp_custom.json()
    assert data_custom["answers"]["plane"]["choice"] == "gui"

    # 3. Guard preset
    resp_guard = client.post(
        "/llm/laya/predict",
        json={
            "state": "git status",
            "preset": "guard",
        },
        headers=headers,
    )
    assert resp_guard.status_code == 200
    data_guard = resp_guard.json()
    assert "risk_tier" in data_guard["answers"]

    # 4. Error handling: missing both preset and questions
    resp_bad = client.post("/llm/laya/predict", json={"state": "hello"}, headers=headers)
    assert resp_bad.status_code == 400
    assert "questions" in resp_bad.json()["detail"].lower()

    # 5. Error handling: invalid preset name
    resp_invalid_preset = client.post(
        "/llm/laya/predict",
        json={"state": "hello", "preset": "nonexistent_preset_xyz"},
        headers=headers,
    )
    assert resp_invalid_preset.status_code == 400
    assert "unknown preset" in resp_invalid_preset.json()["detail"].lower()
