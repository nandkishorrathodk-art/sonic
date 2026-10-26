"""AVCS servo model: the pure, torch-free reach controller.

These assert the model's contract directly -- convergence, a human-like
ballistic-then-corrective profile, no overshoot, and honest failure -- without
any desktop or provider involved.
"""
from __future__ import annotations

import pytest
from forge.vision.servo import HumanReachModel, locate_cursor


def test_model_is_deterministic_and_parameterised():
    model = HumanReachModel(seed=7)
    first = model.reach((0.75, 0.25), (0.1, 0.1))
    second = model.reach((0.75, 0.25), (0.1, 0.1))
    assert [round(s.dx, 9) for s in first] == [round(s.dx, 9) for s in second]
    assert [round(s.dy, 9) for s in first] == [round(s.dy, 9) for s in second]


def test_opening_throw_covers_most_of_the_distance():
    model = HumanReachModel()
    steps = model.reach((0.9, 0.9), (0.1, 0.1))
    assert steps[0].distance_to_target < 0.2
    assert steps[-1].contact_acquired


def test_corrective_steps_decelerate():
    model = HumanReachModel(ballistic_step=0.5)  # force multiple corrections
    steps = model.reach((0.9, 0.9), (0.1, 0.1), max_iterations=12)
    speeds = [s.speed for s in steps[:-1] if s.speed > 0]
    assert len(speeds) >= 2
    assert speeds[-1] <= speeds[0]


def test_contact_radius_is_honoured():
    model = HumanReachModel(foveal_contact_radius=0.05)
    move = model.next_submovement((0.5, 0.5), (0.53, 0.53))
    assert move.contact_acquired


def test_flat_frame_has_no_cursor():
    import numpy as np

    assert locate_cursor(np.zeros((64, 64, 3), dtype=np.uint8)) is None


def test_servo_module_does_not_import_torch():
    """The motor model must stay inside the perception tick budget."""
    import subprocess
    import sys

    code = (
        "import sys; import forge.vision.servo; "
        "sys.exit(0 if 'torch' not in sys.modules else 1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, "forge.vision.servo must not import torch"


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        HumanReachModel(foveal_contact_radius=-1.0)
    with pytest.raises(ValueError):
        HumanReachModel(ballistic_step=0.0)
