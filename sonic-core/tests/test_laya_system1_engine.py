"""Tests for SONIC's Laya System-1 Non-Autoregressive Decision Engine integration."""

from __future__ import annotations

import pytest

from sonic.computer_use.models_boss import SubMission
from sonic.llm.laya_engine import (
    LayaSystem1Engine,
    _detect_script_and_model,
    _has_sufficient_ram_gb,
    get_laya_engine,
)


@pytest.mark.no_live_infra
def test_laya_language_and_checkpoint_routing():
    """Verify English routing to the English Laya checkpoint."""
    lang_en, model_en = _detect_script_and_model("Create a new repository and run pytest")
    assert lang_en == "en"
    assert model_en == "english"

    lang_other, model_other = _detect_script_and_model("Execute terminal command and verify output")
    assert lang_other == "en"
    assert model_other == "english"


@pytest.mark.no_live_infra
def test_laya_english_plane_decision():
    """Verify System-1 `choice` decisions for GUI vs Headless on English goals."""
    engine = LayaSystem1Engine(enable_local_model=False)

    # English headless goal
    plane_hl, prob_hl = engine.decide_execution_plane("Write python code in app.py and run pytest in terminal")
    assert plane_hl == "headless"
    assert prob_hl > 0.55

    # English GUI goal
    plane_gui, prob_gui = engine.decide_execution_plane("Open Chrome browser and click login button on screen")
    assert plane_gui == "gui"
    assert prob_gui > 0.55


@pytest.mark.no_live_infra
def test_laya_submission_plane_inference_english():
    """Verify SubMission._infer_plane_from_goal delegates English tasks to LayaSystem1Engine."""
    sub_headless = SubMission(goal="Write new code in main.py and test it")
    assert sub_headless.plane == "headless"

    sub_gui = SubMission(goal="Open browser window and view user interface")
    assert sub_gui.plane == "gui"


@pytest.mark.no_live_infra
def test_laya_verify_execution_receipt_noul():
    """Verify System-1 `noul` rejects idle dummy terminal text and accepts verified receipts."""
    engine = LayaSystem1Engine(enable_local_model=False)

    dummy_prob = engine.verify_execution_receipt(
        goal="Write a python script",
        evidence="Re-observed: app=, term=Terminal Ready (0 procs)",
    )
    assert dummy_prob < 0.30

    error_prob = engine.verify_execution_receipt(
        goal="Run unit tests",
        evidence="Traceback (most recent call last): AssertionError: 1 failed",
    )
    assert error_prob < 0.30

    valid_prob = engine.verify_execution_receipt(
        goal="Create report.md",
        evidence="FILE_VERIFIED: /home/sonic/workspace/report.md — 200 OK passed",
    )
    assert valid_prob >= 0.65


@pytest.mark.no_live_infra
def test_laya_curiosity_affect_and_ram_guard():
    """Verify BeingMind curiosity affect scoring and laptop RAM safety guard."""
    engine = get_laya_engine()
    res = engine.evaluate_curiosity_affect(
        observation="Discovered new highly novel desktop utility on port 8080",
        known_facts=["Workspace initialized"],
    )
    assert 0.0 <= res["novelty_score"] <= 1.0
    assert 0.0 <= res["worth_learning_prob"] <= 1.0
    assert isinstance(_has_sufficient_ram_gb(0.1), bool)

