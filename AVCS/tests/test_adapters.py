"""Comprehensive Unit Tests for AVCS Universal Ecosystem Adapters."""
from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest
import numpy as np

from adapters.claude_computer_use import AVCSClaudeStreamFilter
from adapters.playwright_eye import AVCSPlaywrightEye
from adapters.openai_operator import AVCSOperatorHarness
from adapters.langchain_tool import AVCSScreenTools
from adapters.cost_benchmark import run_cost_reduction_benchmark
from avcs import SyntheticScreen, ScreenEye


def test_claude_stream_filter():
    """Verify Anthropic Claude Computer Use streaming pre-filter logic."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)
    stream_filter = AVCSClaudeStreamFilter(eye=eye, surprise_threshold=0.3)

    # 1. First frame should always emit
    emit_1, reason_1 = stream_filter.should_emit_screenshot()
    assert emit_1 is True
    assert "Initial" in reason_1
    assert stream_filter.stats.screenshots_sent_to_llm == 1

    # 2. Immediate next frame within throttle window should be suppressed
    emit_2, reason_2 = stream_filter.should_emit_screenshot()
    assert emit_2 is False
    assert stream_filter.stats.frames_suppressed >= 1
    assert stream_filter.stats.estimated_tokens_saved >= 1600

    # 3. Format valid base64 payload
    payload = stream_filter.format_claude_screenshot()
    assert payload["type"] == "image"
    assert payload["source"]["type"] == "base64"
    assert payload["source"]["media_type"] == "image/png"
    assert len(payload["source"]["data"]) > 100


def test_playwright_eye_selector_free_interaction():
    """Verify Playwright adapter executes clicks and types without DOM selectors."""
    mock_page = MagicMock()
    mock_page.viewport_size = {"width": 1920, "height": 1080}
    mock_page.screenshot.return_value = b"\x89PNG\r\n\x1a\n"  # dummy bytes

    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)
    pw_eye = AVCSPlaywrightEye(page=mock_page, eye=eye, viewport_size=(1920, 1080))

    # Visual click
    click_x, click_y = pw_eye.visual_click("Submit Button", click_delay=0.0)
    assert 0 <= click_x <= 1920
    assert 0 <= click_y <= 1080
    assert mock_page.mouse.click.called

    # Visual typing
    mock_page.reset_mock()
    type_x, type_y = pw_eye.visual_type("Search Input", "AVCS Vision")
    assert 0 <= type_x <= 1920
    assert 0 <= type_y <= 1080
    assert mock_page.keyboard.type.called
    mock_page.keyboard.type.assert_called_with("AVCS Vision")


def test_openai_operator_harness():
    """Verify OSWorld / OpenAI Operator gym environment observation harness."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)
    harness = AVCSOperatorHarness(eye=eye)

    obs = harness.process_observation()
    import avcs
    assert obs.screen_category in avcs.UI_CATEGORIES
    assert 0.0 <= obs.category_confidence <= 1.0
    assert isinstance(obs.clickable_buttons, list)
    assert isinstance(obs.typeable_inputs, list)
    assert obs.inference_latency_ms > 0.0

    obs_dict = obs.to_dict()
    assert "screen_category" in obs_dict
    assert "clickable_buttons" in obs_dict

    # Servoing step
    velocity, contact = harness.execute_servoing_step((0.1, 0.1), "Search")
    assert len(velocity) == 2
    assert isinstance(contact, bool)


def test_langchain_screen_tools():
    """Verify LangChain / CrewAI tool definitions and JSON execution."""
    source = SyntheticScreen(width=640, height=360)
    eye = ScreenEye(source=source)
    tools = AVCSScreenTools(eye=eye)

    # 1. look_at_element
    res_look = json.loads(tools.look_at_element("Checkout Button"))
    assert "target_coordinates" in res_look
    assert len(res_look["target_coordinates"]) == 2

    # 2. check_screen_stability
    res_stab = json.loads(tools.check_screen_stability())
    assert "is_stable" in res_stab
    assert "surprise_score" in res_stab

    # 3. get_interactive_buttons
    res_btns = json.loads(tools.get_interactive_buttons())
    assert "clickable_elements" in res_btns

    # 4. get_all_tools schema
    schemas = tools.get_all_tools()
    assert len(schemas) == 4
    tool_names = [s["function"]["name"] for s in schemas]
    assert "look_at_element" in tool_names
    assert "check_screen_stability" in tool_names
    assert "steer_and_click" in tool_names


def test_cost_reduction_benchmark_report():
    """Verify empirical cost and token reduction metrics."""
    report = run_cost_reduction_benchmark(duration_seconds=2.0, fps=10)
    assert report.total_screen_ticks_evaluated == 20
    assert report.baseline_screenshots_sent >= 1
    assert report.avcs_screenshots_sent >= 1
    assert report.tokens_saved >= 0
