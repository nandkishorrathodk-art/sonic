"""Operator CLI for the Laya System-1 decision log.

Usage::

    python -m sonic.llm.decision_log_cli stats
    python -m sonic.llm.decision_log_cli export --out training.jsonl
    python -m sonic.llm.decision_log_cli annotate --record-id <id> \
        --expected '{"plane": "gui"}' --correct

Reading and exporting do not require the recorder to be enabled, so an operator
can inspect and export an existing log without turning recording back on.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sonic.llm.decision_log import DecisionLog


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sonic.llm.decision_log_cli",
        description="Inspect, annotate and export the Laya System-1 decision log.",
    )
    parser.add_argument("--path", default=None, help="log directory (default: env or sonic_data/laya_decisions)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats", help="show counts and log health")

    export = sub.add_parser("export", help="write laya-train compatible JSONL")
    export.add_argument("--out", required=True, help="destination .jsonl path")
    export.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="also export decisions with no observed outcome (empty 'expected')",
    )

    annotate = sub.add_parser("annotate", help="attach an observed outcome to a decision")
    annotate.add_argument("--record-id", required=True)
    annotate.add_argument("--expected", default=None, help="JSON object of ground-truth answers")
    annotate.add_argument("--correct", action="store_true", help="mark the decision as correct")
    annotate.add_argument("--incorrect", action="store_true", help="mark the decision as incorrect")
    annotate.add_argument("--note", default="")
    return parser


def _load_expected(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--expected is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--expected must be a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Reads and writes must work whether or not recording is currently enabled.
    log = DecisionLog(args.path, enabled=True)

    if args.command == "stats":
        print(json.dumps(log.stats(), indent=2))
        return 0

    if args.command == "export":
        summary = log.export_training_jsonl(
            args.out, require_labels=not args.include_unlabelled
        )
        print(json.dumps(summary, indent=2))
        if summary.get("written", 0) == 0:
            print(
                "No labelled decisions were exported. Annotate decisions with "
                "`annotate` before training; an unlabelled row is not training data.",
                file=sys.stderr,
            )
        return 0

    if args.command == "annotate":
        if args.correct and args.incorrect:
            raise SystemExit("--correct and --incorrect are mutually exclusive")
        correct: bool | None = None
        if args.correct:
            correct = True
        elif args.incorrect:
            correct = False
        stored = log.record_outcome(
            args.record_id,
            expected=_load_expected(args.expected),
            correct=correct,
            note=args.note,
        )
        if not stored:
            print("Outcome was not stored.", file=sys.stderr)
            return 1
        print(f"Recorded outcome for {args.record_id}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
