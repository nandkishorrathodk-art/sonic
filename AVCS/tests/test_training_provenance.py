import pytest

from avcs.training.train_kaggle import build_parser, main


def test_synthetic_trainer_requires_explicit_acknowledgment(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    assert "synthetic colored blocks only" in capsys.readouterr().err


def test_synthetic_trainer_defaults_to_separate_output_directory():
    args = build_parser().parse_args(["--synthetic-data-only"])

    assert args.output_dir == "runs/synthetic_cortex_2m"
