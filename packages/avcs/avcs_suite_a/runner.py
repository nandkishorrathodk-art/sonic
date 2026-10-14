"""CLI runner for AVCS Suite A.

An adapter is project-specific because this bundle intentionally does not assume
an undocumented AVCS SDK API. Example:
    AVCS_SUITE_ADAPTER=my_project.avcs_adapter:create_adapter python -m avcs_suite_a.runner
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from .catalog import CASES, CASE_BY_ID, validate_catalog
from .evaluator import score_case, summarize
from .models import AgentAdapter, Rect


def _request(url: str, *, token: str | None = None, method: str = "GET",
             payload: Mapping[str, Any] | None = None, timeout: float = 3.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_adapter(spec: str) -> AgentAdapter:
    if ":" not in spec:
        raise ValueError("adapter must use module.path:factory_name syntax")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    adapter = factory()
    if not callable(getattr(adapter, "execute", None)):
        raise TypeError("adapter object must implement execute(task, timeout_s)")
    return adapter


def _wait_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float = 20) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not started"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"fixture UI exited early (code {proc.returncode}); stdout={proc.stdout.read() if proc.stdout else ''}")
        try:
            if _request(base_url + "/health").get("ready"):
                return
        except Exception as exc:  # startup race; report only after deadline
            last_error = repr(exc)
        time.sleep(0.1)
    raise TimeoutError(f"fixture UI did not become ready: {last_error}")


def _wait_for_state(base_url: str, token: str, case, timeout_s: float = 2.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _request(base_url + "/oracle", token=token)
        state = last.get("state", {})
        if all(state.get(k) == v for k, v in case.expected_state.items()):
            return last
        time.sleep(0.05)
    return last


def run_suite(adapter: AgentAdapter, base_url: str, token: str, case_ids: list[str] | None = None) -> dict[str, Any]:
    selected = [CASE_BY_ID[cid] for cid in case_ids] if case_ids else list(CASES)
    scores = []
    trace: list[dict[str, Any]] = []
    for case in selected:
        record: dict[str, Any] = {"case_id": case.id, "started_at_monotonic": time.monotonic()}
        try:
            _request(base_url + "/admin/reset", token=token, method="POST", payload={"case_id": case.id})
            before = _request(base_url + "/oracle", token=token)
            target_map = before.get("target_boxes", {})
            target_rect = Rect.from_value(target_map.get(case.target_id)) if case.target_id else None
            start = time.monotonic()
            try:
                run_result = adapter.execute(case.task, case.timeout_s)
                run_map = dict(run_result) if isinstance(run_result, Mapping) else {"status": "failed", "error": "adapter_return_not_mapping"}
            except TimeoutError as exc:
                run_map = {"status": "failed", "error": f"adapter_timeout: {exc}"}
            except Exception as exc:
                run_map = {"status": "failed", "error": f"adapter_exception: {type(exc).__name__}: {exc}"}
            run_map.setdefault("duration_ms", (time.monotonic() - start) * 1000.0)
            after = _wait_for_state(base_url, token, case, timeout_s=2.5)
            score = score_case(case, before, after, target_rect, run_map)
            scores.append(score)
            record.update({"run": run_map, "before": before, "after": after, "score": score.to_dict()})
        except Exception as exc:
            from .models import CaseScore
            score = CaseScore(case_id=case.id, category=case.category, expected_status=case.expected_status,
                              reported_status=None, task_success=False, expected_state_match=False,
                              target_scored=False, target_iou=None, target_match=None,
                              false_confirmation=False, elapsed_ms=None,
                              error=f"harness_exception: {type(exc).__name__}: {exc}")
            scores.append(score)
            record["score"] = score.to_dict()
        trace.append(record)
    summary = summarize(scores)
    summary["suite_name"] = "AVCS Suite A — deterministic Windows fixtures"
    summary["environment"] = {
        "python": sys.version.split()[0], "platform": sys.platform,
        "adapter": os.environ.get("AVCS_SUITE_ADAPTER", "<passed-in adapter>"),
    }
    summary["trace"] = trace
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic desktop fixtures against a real AVCS adapter.")
    parser.add_argument("--adapter", default=os.environ.get("AVCS_SUITE_ADAPTER"),
                        help="Project adapter module:factory; it must drive the visible AVCS UI.")
    parser.add_argument("--case", action="append", dest="cases", help="Case ID (repeat for multiple). Defaults to all cases.")
    parser.add_argument("--output", default="suite_a_results.json", help="JSON output path")
    parser.add_argument("--no-ui", action="store_true", help="Do not start fixture UI; use an already-running fixture server.")
    parser.add_argument("--base-url", default=None, help="Existing fixture server URL, e.g. http://127.0.0.1:8765")
    args = parser.parse_args()

    errors = validate_catalog()
    if errors:
        print("Invalid fixture catalog:\n" + "\n".join(errors), file=sys.stderr)
        return 2
    if not args.adapter:
        print("No AVCS adapter configured. Set AVCS_SUITE_ADAPTER=module:factory or pass --adapter.\n"
              "The fixture app and independent scorer can be tested without this adapter, but end-to-end results require it.",
              file=sys.stderr)
        return 2
    try:
        adapter = _load_adapter(args.adapter)
    except Exception as exc:
        print(f"Could not load adapter: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    token = secrets.token_urlsafe(32)
    proc: subprocess.Popen[str] | None = None
    base_url = args.base_url
    if not args.no_ui:
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        os.environ["AVCS_FIXTURE_URL"] = base_url
        child_env = dict(os.environ)
        child_env["AVCS_FIXTURE_ORACLE_TOKEN"] = token
        proc = subprocess.Popen([sys.executable, "-m", "avcs_suite_a.fixture_app", "--port", str(port)],
                                env=child_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        try:
            _wait_health(base_url, proc)
        except Exception:
            proc.terminate()
            raise
    elif not base_url:
        print("--no-ui requires --base-url", file=sys.stderr)
        return 2
    else:
        # In external-UI mode token must also be configured for the fixture server.
        supplied = os.environ.get("AVCS_FIXTURE_ORACLE_TOKEN")
        if not supplied:
            print("--no-ui requires AVCS_FIXTURE_ORACLE_TOKEN for the running fixture server", file=sys.stderr)
            return 2
        token = supplied

    try:
        report = run_suite(adapter, base_url, token, args.cases)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in report if k not in {"results", "trace"}}, indent=2))
        print(f"Full report: {out_path.resolve()}")
        return 0 if report.get("suite_case_pass_rate") == 1.0 else 1
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
