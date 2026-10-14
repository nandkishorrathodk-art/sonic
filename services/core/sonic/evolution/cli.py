"""
SONIC v2 — Self-Evolution Engine CLI Runner
===========================================
Dedicated command-line interface for running automated codebase upgrades,
bug fixes, and logical enhancements.

Usage Examples:
    # 1. Submit an intelligent goal:
    python -m sonic.evolution.cli goal --title "Optimize parser latency" --desc "Use bisect" --category logic_improvement --execute

    # 2. List queued and active goals:
    python -m sonic.evolution.cli goals

    # 3. Run continuous evolution on queued goals:
    python -m sonic.evolution.cli run --max-goals 5

    # 4. View current semantic version and history:
    python -m sonic.evolution.cli version

    # 5. View live evolution journal:
    python -m sonic.evolution.cli journal

    # 6. Verify and record a direct diff patch without applying it:
    python -m sonic.evolution.cli patch --target services/core/sonic/tools/sample.py --diff-file fix.patch --desc "Fix bug"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from sonic.evolution.codebase_evolver import CodebaseEvolver, EvolutionSummaryReport
from sonic.evolution.evolution_journal import EvolutionJournal
from sonic.evolution.goal_director import EvolutionGoalDirector, GoalStatus
from sonic.evolution.pipeline import EvolutionStage
from sonic.evolution.version_tracker import VersionTracker


def is_safe_repo_path(path_str: str | Path | None, repo_root: Path) -> bool:
    """Verify that path_str does not escape repo_root via directory traversal."""
    if not path_str or not str(path_str).strip():
        return False
    p = Path(path_str)
    root = repo_root.resolve()
    resolved = p.resolve() if p.is_absolute() else (root / p).resolve()
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def _positive_int(val: str) -> int:
    try:
        ival = int(val)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid integer: '{val}'")
    if ival <= 0:
        raise argparse.ArgumentTypeError(f"--max-goals must be greater than 0 (got {ival})")
    return ival


def generate_changelog(
    self: Any = None,
    from_version: str | None = None,
    to_version: str | None = None,
) -> str:
    """Generate a clean Markdown changelog between versions.

    Delegates to EvolutionJournal.generate_changelog with version range filtering.
    """
    if isinstance(self, EvolutionJournal):
        return self.generate_changelog(from_version=from_version, to_version=to_version)
    if isinstance(self, str):
        to_version = from_version
        from_version = self
    journal = EvolutionJournal()
    return journal.generate_changelog(from_version=from_version, to_version=to_version)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sonic.evolution.cli",
        description="🧬 SONIC Self-Evolution Engine — Autonomous Codebase Evolution Director",
    )
    subparsers = parser.add_subparsers(dest="command", help="Evolution command")

    # Command: goal
    goal_parser = subparsers.add_parser("goal", help="Enqueue and optionally verify an evolution goal")
    goal_parser.add_argument("--title", required=True, help="Title of the evolution goal or bug fix")
    goal_parser.add_argument("--desc", required=True, help="Detailed description of what to repair or improve")
    goal_parser.add_argument("--priority", type=int, default=2, choices=[1, 2, 3, 4], help="Priority (1=critical, 2=high, 3=medium, 4=low)")
    goal_parser.add_argument("--category", default="logic_improvement", choices=["bug_fix", "logic_improvement", "capability_add", "performance", "refactor"], help="Goal category")
    goal_parser.add_argument("--targets", nargs="*", help="Optional target file paths relative to repo root")
    goal_parser.add_argument("--execute", action="store_true", help="Immediately execute the goal after submitting")
    goal_parser.add_argument("--auto-promote", action="store_true", default=False, help="Explicitly apply and commit a verified goal")
    goal_parser.add_argument("--push", action="store_true", default=False, help="Push a promoted goal to origin (requires --auto-promote)")

    # Command: goals
    goals_parser = subparsers.add_parser("goals", help="List all evolution goals in the queue")
    goals_parser.add_argument("--status", choices=["queued", "analyzing", "synthesizing", "testing", "verified", "awaiting_approval", "promoted", "rolled_back", "rejected", "failed"], help="Filter by goal status")

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Approve or reject a verified goal awaiting operator resolution",
    )
    resolve_parser.add_argument("goal_id", help="Goal ID in awaiting_approval state")
    resolve_parser.add_argument(
        "--decision",
        required=True,
        choices=["approve", "reject"],
        help="Approve the verified working-tree patch or reject it without discarding edits",
    )
    resolve_parser.add_argument(
        "--push",
        action="store_true",
        default=False,
        help="Push the newly verified local promotion to origin (approval only)",
    )

    # Command: run
    run_parser = subparsers.add_parser("run", help="Verify and record queued evolution goals")
    run_parser.add_argument("--max-goals", type=_positive_int, default=None, help="Maximum number of goals to process (must be > 0)")
    run_parser.add_argument("--auto-promote", action="store_true", default=False, help="Explicitly apply and commit verified goals")
    run_parser.add_argument("--push", action="store_true", default=False, help="Push promoted goals to origin (requires --auto-promote)")

    # Command: version
    subparsers.add_parser("version", help="Show current semantic version milestone and advancement history")

    # Command: journal
    journal_parser = subparsers.add_parser("journal", help="Show recent evolution journal entries and fitness metrics")
    journal_parser.add_argument("--limit", type=int, default=15, help="Number of entries to show")

    # Command: changelog
    changelog_parser = subparsers.add_parser("changelog", help="Print evolution changelog between versions")
    changelog_parser.add_argument("--from-ver", help="Starting version")
    changelog_parser.add_argument("--to-ver", help="Ending version")

    # Command: patch
    patch_parser = subparsers.add_parser("patch", help="Verify and record a code patch without applying it")
    patch_parser.add_argument("--target", required=True, help="Target component file relative to repository root")
    patch_parser.add_argument("--desc", required=True, help="Description of the bug fix, upgrade, or logical enhancement")
    patch_parser.add_argument("--diff-file", help="Path to a unified diff (.patch) or code replacement file")
    patch_parser.add_argument("--diff", help="Raw diff or code string passed directly")
    patch_parser.add_argument("--auto-promote", action="store_true", default=False, help="Explicitly apply and commit after verification")
    patch_parser.add_argument("--push", action="store_true", default=False, help="Push after promotion (requires --auto-promote)")
    patch_parser.add_argument("--tests", nargs="*", help="Specific pytest files/paths to verify against")
    patch_parser.add_argument("--summary-out", help="Optional output path to save the generated markdown summary")

    # Command: status
    subparsers.add_parser("status", help="Show self-evolution engine safety status, protected components, and evolution.md path")

    selfcoder_parser = subparsers.add_parser("selfcode", help="Inspect SONIC's own source and verify fixes (report-only by default)")
    selfcoder_parser.add_argument("--cycles", type=int, default=1, help="Maximum improvement cycles to run")
    selfcoder_parser.add_argument("--push", action="store_true", default=False, help="Deprecated; self-coder never commits or pushes")
    selfcoder_parser.add_argument("--write", action="store_true", default=False, help="Apply verified changes in-place after isolated tests (no commit)")

    audit_parser = subparsers.add_parser("audit", help="Read-only full-codebase check: health, defects, failing tests")
    audit_parser.add_argument("--no-tests", action="store_true", default=False, help="Skip running the test suite")
    audit_parser.add_argument("--limit", type=int, default=60, help="Maximum findings to report")

    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    evolver = CodebaseEvolver()
    version_tracker = VersionTracker(repo_root=evolver.repo_root)
    journal = EvolutionJournal(repo_root=evolver.repo_root)
    director = EvolutionGoalDirector(
        repo_root=evolver.repo_root,
        evolver=evolver,
        version_tracker=version_tracker,
        journal=journal,
    )

    if args.command == "audit":
        from sonic.evolution.codebase_auditor import CodebaseAuditor
        from sonic.evolution.codebase_report import narrate

        auditor = CodebaseAuditor(repo_root=evolver.repo_root, max_findings=max(1, args.limit))
        audit_report = auditor.audit(run_tests=not args.no_tests)
        print(narrate(audit_report))
        print()
        print(audit_report.to_text())
        return 0

    if args.command == "selfcode":
        import asyncio as _asyncio

        from sonic.evolution.codebase_auditor import CodebaseAuditor
        from sonic.evolution.codebase_report import narrate
        from sonic.evolution.self_coder import SelfCoder

        if args.push:
            print("Self-coder verified updates never commit or push.")
            return 2

        # Always look at the codebase first, then propose/verify fixes.
        auditor = CodebaseAuditor(repo_root=evolver.repo_root)
        audit_report = auditor.audit(run_tests=False)
        coder = SelfCoder(repo_root=evolver.repo_root, evolver=evolver, director=director,
                          max_cycles=max(1, args.cycles), push=args.push,
                          allow_writes=bool(args.write))
        report = _asyncio.run(coder.run(max_cycles=max(1, args.cycles)))
        print(report.to_text())
        print()
        print(narrate(audit_report, cycles=report.cycles))
        return 0

    if args.command == "status":
        print("=" * 70)
        print("🧬 SONIC CONTINUOUS CODEBASE EVOLUTION ENGINE")
        print("=" * 70)
        print(f"Repository Root:    {evolver.repo_root}")
        print(f"Current Version:    v{version_tracker.current_version()}")
        print(f"Live Notes File:    {journal.evolution_md_path}")
        print(f"Queued Goals:       {len(director.list_goals(status=GoalStatus.QUEUED))}")
        metrics = journal.fitness_metrics()
        print(f"Total Cycles:       {metrics['total_cycles']} (Promoted: {metrics['promoted']}, Rollbacks: {metrics['rolled_back']})")
        print(f"Evolution Fitness:  {metrics['success_rate'] * 100:.1f}%")
        print("\nProtected Safety Components (AI Self-Modification FORBIDDEN):")
        from sonic.evolution.codebase_evolver import _PROTECTED_COMPONENTS
        for c in sorted(_PROTECTED_COMPONENTS):
            print(f"  ⛔ {c}")
        print("\nAll other components (tools, agents, research, mission_engine) are EVOLVABLE.")
        return 0

    if args.command == "version":
        current = version_tracker.current_version()
        hist = version_tracker.history()
        print(f"\n🧬 Current SONIC Version: v{current}")
        print("-" * 50)
        print(f"Milestone History ({len(hist)} versions):")
        for v in hist:
            tag_str = f" [{v.git_tag}]" if v.git_tag else ""
            print(f"  • v{v.version}{tag_str} ({v.timestamp[:19]}) — {v.title} ({v.bump_type})")
        print()
        return 0

    if args.command == "journal":
        entries = journal.get_entries(limit=args.limit)
        metrics = journal.fitness_metrics()
        print(f"\n📜 SONIC Evolution Journal (Fitness: {metrics['success_rate'] * 100:.1f}% | Total: {metrics['total_cycles']})")
        print(f"Notes file: {journal.evolution_md_path}\n" + "-" * 70)
        for e in entries:
            st_badge = "🟢" if e.status == "promoted" else ("🔴" if e.status == "rolled_back" else "⛔")
            print(f"{st_badge} [{e.timestamp[:19]}] {e.title} (v{e.version_before} -> v{e.version_after})")
            print(f"   Category: {e.category} | Impact: +{e.lines_added}/-{e.lines_removed} lines")
            if e.commit_hash:
                print(f"   Commit: {e.commit_hash}")
        print()
        return 0

    if args.command == "changelog":
        print(journal.generate_changelog(args.from_ver, args.to_ver))
        return 0

    if args.command == "goal":
        if (args.push or args.auto_promote) and not args.execute:
            print("Error: --auto-promote and --push require --execute", file=sys.stderr)
            return 1
        if args.push and not args.auto_promote:
            print("Error: --push requires --auto-promote", file=sys.stderr)
            return 1

        goal = director.submit_goal(
            title=args.title,
            description=args.desc,
            priority=args.priority,
            category=args.category,
            target_files=args.targets,
        )
        print(f"✅ Submitted Evolution Goal: `{goal.goal_id}`")
        print(f"   Title:    {goal.title}")
        print(f"   Category: {goal.category.value}")
        print(f"   Priority: {goal.priority}")

        if args.execute:
            print("\n🧬 Executing Goal Now...")
            import asyncio as _asyncio

            rep = _asyncio.run(
                director.execute_goal(
                    goal.goal_id,
                    auto_promote=args.auto_promote,
                    push=args.push,
                    report_only=not args.auto_promote,
                )
            )
            print(f"Outcome: {rep.stage.value}")
            print(f"New Version: v{version_tracker.current_version()}")
            if args.auto_promote:
                print(f"Notes updated in: {journal.evolution_md_path}")
            else:
                print("Verified diff and test receipts were recorded on the goal; active checkout unchanged.")
            return 0 if rep.stage in {
                EvolutionStage.PROMOTED,
                EvolutionStage.AWAITING_APPROVAL,
            } else 2
        return 0

    if args.command == "goals":
        goals = director.list_goals(status=args.status)
        print(f"\n📋 SONIC Evolution Goals ({len(goals)}):")
        print("-" * 70)
        for g in goals:
            status_icon = "🟢" if g.status == GoalStatus.PROMOTED else ("⏳" if g.status == GoalStatus.QUEUED else "⚠️")
            print(f"{status_icon} [{g.goal_id}] (P{g.priority}) {g.title} [{g.status.value}]")
            print(f"   Category: {g.category.value} | Created: {g.created_at[:19]}")
            if g.version_after:
                print(f"   Result: v{g.version_before} -> v{g.version_after}")
        print()
        return 0

    if args.command == "resolve":
        if args.push and args.decision != "approve":
            print("Error: --push is valid only with --decision approve", file=sys.stderr)
            return 1
        try:
            if args.decision == "reject":
                goal = director.reject_goal(args.goal_id)
                print(f"Goal {goal.goal_id} rejected.")
                print(goal.notes)
                return 0
            report = director.approve_goal(args.goal_id, push=args.push)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Outcome: {report.stage.value}")
        if report.git_commit_hash:
            print(f"Commit: {report.git_commit_hash}")
        if report.error_reason:
            print(report.error_reason)
        goal = director.get_goal(args.goal_id)
        if goal is not None and goal.notes:
            print(goal.notes)
        return 0 if report.stage == EvolutionStage.PROMOTED else 2

    if args.command == "run":
        if args.max_goals is not None and args.max_goals <= 0:
            print("Error: --max-goals must be greater than 0", file=sys.stderr)
            return 1
        if args.push and not args.auto_promote:
            print("Error: --push requires --auto-promote", file=sys.stderr)
            return 1
        print(f"🧬 Starting continuous evolution loop (max_goals={args.max_goals})...")
        import asyncio as _asyncio

        reps = _asyncio.run(
            director.run_continuous(
                max_goals=args.max_goals,
                push=args.push,
                auto_promote=args.auto_promote,
                report_only=not args.auto_promote,
            )
        )
        print(f"\n✅ Finished continuous run. Processed {len(reps)} goals.")
        print(f"Active Version: v{version_tracker.current_version()}")
        if args.auto_promote:
            print(f"All notes logged to: {journal.evolution_md_path}")
        else:
            print("Verified diffs and test receipts were recorded on their goals; active checkout unchanged.")
        return 0

    if args.command == "patch":
        if args.push and not args.auto_promote:
            print("Error: --push requires --auto-promote", file=sys.stderr)
            return 1
        repo_root = evolver.repo_root.resolve()

        if not is_safe_repo_path(args.target, repo_root):
            print(f"Error: Target path escapes repository root: {args.target}", file=sys.stderr)
            return 1

        if args.summary_out and not is_safe_repo_path(args.summary_out, repo_root):
            print(f"Error: Summary output path escapes repository root: {args.summary_out}", file=sys.stderr)
            return 1

        diff_content = ""
        if args.diff_file:
            p = Path(args.diff_file)
            diff_p = p.resolve() if p.is_absolute() else (repo_root / args.diff_file).resolve()
            if not diff_p.exists():
                print(f"Error: Diff file not found: {args.diff_file}", file=sys.stderr)
                return 1
            diff_content = diff_p.read_text(encoding="utf-8", errors="replace")
        elif args.diff:
            diff_content = args.diff
        else:
            print("Error: Must provide either --diff-file or --diff", file=sys.stderr)
            return 1

        print("🧬 Starting Codebase Evolution Cycle...")
        print(f"   Target:      {args.target}")
        print(f"   Description: {args.desc}")
        print(f"   Auto-Promote:{args.auto_promote}")
        print(f"   Git Push:    {args.push}")
        print("-" * 70)

        report: EvolutionSummaryReport = evolver.evolve(
            target_component=args.target,
            description=args.desc,
            code_diff=diff_content,
            auto_promote=args.auto_promote,
            push=args.push,
            test_paths=args.tests,
            report_only=not args.auto_promote,
        )

        md = report.to_markdown()
        print("\n" + md + "\n")

        if args.summary_out:
            p = Path(args.summary_out)
            out_p = p.resolve() if p.is_absolute() else (repo_root / args.summary_out).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(md, encoding="utf-8")
            print(f"📝 Summary saved to: {out_p}")

        if args.auto_promote:
            print(f"📝 Live notes appended to: {journal.evolution_md_path}")
        else:
            print("📝 Report contains the verified proposal diff; active checkout unchanged.")

        if report.stage in (EvolutionStage.PROMOTED, EvolutionStage.AWAITING_APPROVAL):
            return 0
        else:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
