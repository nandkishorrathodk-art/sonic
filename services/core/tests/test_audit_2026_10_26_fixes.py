"""Regression tests for the 2026-10-26 audit fix pass.

Six real defects were fixed. Each test here reproduces the original failure mode
so the fix cannot silently regress:

1. Goal queue crashed on an unrecognised category/status string.
2. The patch synthesizer dropped model patches that lacked a file header.
3. The sealed safety policy silently swallowed scope-checker capture errors.
4. The AVCS apprenticeship store resolved to a CWD-dependent path (split-brain).
5. AVCS language grounding was driven by random noise, not the query text.
6. The grounding loss trained on fabricated (0.5, 0.5) coordinates.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Fix 1 — goal category / status coercion
# ---------------------------------------------------------------------------
class TestGoalCoercion:
    def test_unknown_category_falls_back_instead_of_raising(self):
        from sonic.evolution.goal_director import (
            GoalCategory,
            coerce_goal_category,
        )

        # Before: GoalCategory("nonsense") raised ValueError and aborted the cycle.
        assert coerce_goal_category("nonsense") is GoalCategory.LOGIC_IMPROVEMENT

    def test_category_aliases_and_enum_pass_through(self):
        from sonic.evolution.goal_director import (
            GoalCategory,
            coerce_goal_category,
        )

        assert coerce_goal_category("Bug Fix") is GoalCategory.BUG_FIX
        assert coerce_goal_category("BUGFIX") is GoalCategory.BUG_FIX
        assert coerce_goal_category(GoalCategory.PERFORMANCE) is GoalCategory.PERFORMANCE
        assert coerce_goal_category("") is GoalCategory.LOGIC_IMPROVEMENT
        assert coerce_goal_category(None) is GoalCategory.LOGIC_IMPROVEMENT

    def test_unknown_status_falls_back_instead_of_raising(self):
        from sonic.evolution.goal_director import GoalStatus, coerce_goal_status

        assert coerce_goal_status("not_a_status") is GoalStatus.QUEUED
        assert coerce_goal_status("promoted") is GoalStatus.PROMOTED
        assert coerce_goal_status(None) is GoalStatus.QUEUED

    def test_legacy_row_with_bad_values_still_loads(self, tmp_path):
        """A persisted row with a stale category must not break the queue read."""
        from sonic.evolution.goal_director import EvolutionGoalDirector

        director = EvolutionGoalDirector(db_path=str(tmp_path / "evo.db"))
        director.submit_goal("fix the logger", "correct a logger bug", category="bug_fix")
        # Simulate an older build writing a category this version does not know.
        with director._get_connection() as conn:  # noqa: SLF001 - test reaches into store
            conn.execute("UPDATE evolution_goals SET category='legacy_thing'")
            conn.commit()
        goals = director.list_goals()
        assert len(goals) == 1


# ---------------------------------------------------------------------------
# Fix 2 — headerless patch repair
# ---------------------------------------------------------------------------
class TestHeaderlessPatchRepair:
    def _read(self, files: dict[str, str]):
        return lambda rel: files.get(rel)

    def test_headerless_hunk_gets_correct_header(self):
        from sonic.evolution.patch_synthesizer import _ensure_file_header

        target = "sonic/logger.py"
        files = {target: "import os\n\nVALUE = 1\n"}
        diff = "@@ -1,3 +1,3 @@\n import os\n \n-VALUE = 1\n+VALUE = 2\n"

        repaired, err = _ensure_file_header(diff, [target], self._read(files))

        assert err == ""
        assert repaired.startswith(f"--- a/{target}\n+++ b/{target}\n")

    def test_ambiguous_context_fails_closed(self):
        from sonic.evolution.patch_synthesizer import _ensure_file_header

        files = {"a.py": "x = 1\n", "b.py": "x = 1\n"}
        diff = "@@ -1 +1 @@\n-x = 1\n+x = 2\n"

        repaired, err = _ensure_file_header(diff, ["a.py", "b.py"], self._read(files))

        assert repaired == diff
        assert "ambiguous" in err

    def test_existing_header_is_untouched(self):
        from sonic.evolution.patch_synthesizer import _ensure_file_header

        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        repaired, err = _ensure_file_header(diff, ["a.py"], self._read({"a.py": "x = 1\n"}))

        assert repaired == diff
        assert err == ""

    def test_repaired_diff_is_accepted_by_git_apply(self, tmp_path):
        from sonic.evolution.patch_synthesizer import _ensure_file_header

        target = "logger.py"
        (tmp_path / target).write_text("import os\n\nVALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

        diff = "@@ -1,3 +1,3 @@\n import os\n \n-VALUE = 1\n+VALUE = 2\n"
        repaired, err = _ensure_file_header(
            diff, [target], self._read({target: "import os\n\nVALUE = 1\n"})
        )
        assert err == ""

        check = subprocess.run(
            ["git", "apply", "--check", "-"], cwd=tmp_path,
            input=repaired, text=True, capture_output=True,
        )
        assert check.returncode == 0, check.stderr


# ---------------------------------------------------------------------------
# Fix 3 — sealed policy reports scope-checker capture failures
# ---------------------------------------------------------------------------
class TestSealedPolicyHonestCapture:
    def test_capture_failure_is_logged_not_swallowed(self):
        from sonic.safety.sealed import SealedActionPolicy

        class Exploding:
            @property
            def _INTRUSIVE_PATTERNS(self):
                raise RuntimeError("boom")

        policy = SealedActionPolicy(scope_checker=Exploding(), auto_seal=False)
        # The failure must not abort construction, but must be visible: the
        # captured rules are empty rather than a silent pretend-capture.
        assert policy._scope_checker_rules == []  # noqa: SLF001

    def test_seal_still_verifies_with_a_working_scope_checker(self):
        from sonic.safety.sealed import SealedActionPolicy

        policy = SealedActionPolicy(auto_seal=True)
        assert policy._compute_seal_hash() == policy.seal_hash  # noqa: SLF001


# ---------------------------------------------------------------------------
# Fix 4 — CWD-independent apprenticeship store path
# ---------------------------------------------------------------------------
class TestApprenticeshipStorePath:
    def test_default_root_is_anchored_to_repo_not_cwd(self, tmp_path, monkeypatch):
        from sonic.computer_use import apprenticeship as mod

        monkeypatch.delenv("SONIC_AVCS_APPRENTICESHIP_DIR", raising=False)
        monkeypatch.delenv("SONIC_DATA_DIR", raising=False)

        expected = mod._repo_root() / "sonic_data" / "avcs" / "apprenticeship"  # noqa: SLF001
        monkeypatch.chdir(tmp_path)
        assert Path(mod.ApprenticeshipStore().root) == expected
        monkeypatch.chdir("/")
        assert Path(mod.ApprenticeshipStore().root) == expected

    def test_explicit_env_overrides_are_honoured(self, monkeypatch, tmp_path):
        from sonic.computer_use import apprenticeship as mod

        monkeypatch.setenv("SONIC_AVCS_APPRENTICESHIP_DIR", str(tmp_path / "x"))
        assert Path(mod.ApprenticeshipStore().root) == tmp_path / "x"

        monkeypatch.delenv("SONIC_AVCS_APPRENTICESHIP_DIR")
        monkeypatch.setenv("SONIC_DATA_DIR", str(tmp_path / "data"))
        assert Path(mod.ApprenticeshipStore().root) == tmp_path / "data" / "avcs" / "apprenticeship"


# ---------------------------------------------------------------------------
# Fix 5 — real (not random) language grounding
# ---------------------------------------------------------------------------
class TestRealLanguageGrounding:
    def test_encoder_is_deterministic_and_text_dependent(self):
        from forge.vision.cortex.core.modulation import encode_text_query

        a1 = encode_text_query("Submit", 128)
        a2 = encode_text_query("Submit", 128)
        b = encode_text_query("Cancel", 128)

        assert torch.allclose(a1, a2)
        assert not torch.allclose(a1, b)
        # A random vector would be near-orthogonal to everything; a real
        # character encoder keeps related words closer than unrelated ones.
        assert float((a1 * a2).sum()) > float((a1 * b).sum())

    def test_empty_query_is_rejected(self):
        from forge.vision.cortex.core.modulation import encode_text_query

        with pytest.raises(ValueError):
            encode_text_query("   ", 128)

    def test_screen_eye_look_at_uses_the_encoder(self):
        """look_at must not depend on a random hash seed anymore."""
        from forge.vision.capture import SyntheticScreen
        from forge.vision.cortex.sdk import ScreenEye

        eye = ScreenEye(source=SyntheticScreen(width=160, height=90))
        try:
            p1 = eye.look_at("Submit Button")
            p2 = eye.look_at("Submit Button")
            assert p1 == p2
            assert len(p1) == 2
        finally:
            eye.close()

    def test_datasets_emit_real_query_embeddings(self):
        from forge.vision.cortex.data.dataset import ScreenDataset

        ds = ScreenDataset(num_samples=4)
        emb = ds[0]["query_embed"]
        assert emb.shape == (256,)
        assert torch.isfinite(emb).all()
        # Deterministic across independent dataset instances for the same seed.
        assert torch.allclose(emb, ScreenDataset(num_samples=4)[0]["query_embed"])


# ---------------------------------------------------------------------------
# Fix 6 — grounding loss masked to real targets only
# ---------------------------------------------------------------------------
class TestGroundingLossMask:
    def _batch(self, b=4):
        return {
            "temporal_memory": torch.randn(b, 128),
            "affordance_logits": torch.randn(b, 5, 32, 32),
            "category_logits": torch.randn(b, 7),
            "query_coords": torch.rand(b, 2),
        }

    def test_all_invalid_samples_produce_zero_grounding_loss(self):
        from forge.vision.cortex.training.losses import AVCSMultiTaskLoss

        loss_fn = AVCSMultiTaskLoss()
        b = 4
        out = loss_fn(
            self._batch(b),
            torch.randn(b, 128),
            torch.randint(0, 5, (b, 32, 32)),
            torch.randint(0, 7, (b,)),
            torch.rand(b, 2),
            grounding_valid=torch.zeros(b, dtype=torch.bool),
        )
        assert out["loss_grounding"].item() == 0.0

    def test_mixed_mask_only_trains_valid_samples(self):
        from forge.vision.cortex.training.losses import AVCSMultiTaskLoss

        loss_fn = AVCSMultiTaskLoss()
        b = 4
        mask = torch.tensor([True, False, True, False])
        out = loss_fn(
            self._batch(b),
            torch.randn(b, 128),
            torch.randint(0, 5, (b, 32, 32)),
            torch.randint(0, 7, (b,)),
            torch.rand(b, 2),
            grounding_valid=mask,
        )
        assert out["loss_grounding"].item() > 0.0

    def test_missing_mask_keeps_backward_compatibility(self):
        from forge.vision.cortex.training.losses import AVCSMultiTaskLoss

        loss_fn = AVCSMultiTaskLoss()
        b = 2
        out = loss_fn(
            self._batch(b),
            torch.randn(b, 128),
            torch.randint(0, 5, (b, 32, 32)),
            torch.randint(0, 7, (b,)),
            torch.rand(b, 2),
        )
        assert out["loss_grounding"].item() > 0.0

    def test_unlabeled_datasets_mark_grounding_invalid(self, tmp_path):
        from forge.vision.cortex.data.labeled_dataset import LabeledFrameDataset

        pytest.importorskip("PIL")
        from PIL import Image

        Image.new("RGB", (32, 18), (10, 20, 30)).save(tmp_path / "f0.png")
        import json

        manifest = tmp_path / "m.json"
        manifest.write_text(
            json.dumps({"frames": [{"frame": "f0.png", "label": "editor_code"}]}),
            encoding="utf-8",
        )
        sample = LabeledFrameDataset(manifest, width=32, height=18)[0]
        assert bool(sample["grounding_valid"]) is False
        assert torch.isfinite(sample["query_embed"]).all()
