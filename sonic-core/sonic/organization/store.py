"""Tenant-scoped persistent organization registry."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sonic.logger import get_logger
from sonic.organization.architecture import DEPARTMENTS, SUBSYSTEM_OWNERSHIP, TEAMS

logger = get_logger(__name__)

_DEPARTMENTS = tuple(
    (department.value, department.value.replace("_", " ").title(), "")
    for department in DEPARTMENTS
)
_TEAMS = tuple(
    (team.name.value, team.name.value.replace("_", " ").title(), team.department.value)
    for team in TEAMS
)

_SEED_EMPLOYEES = (
    ("cognitive-lead", "Cognitive Systems Lead", "cognitive_systems", "reasoning_orchestration"),
    ("computer-lead", "Computer & Interaction Lead", "computer_interaction", "desktop_control"),
    ("platform-lead", "Platform & Systems Lead", "platform_systems", "api_services"),
    ("safety-lead", "Security & Safety Lead", "security_safety", "trust_safety"),
    ("evaluation-lead", "Evaluation & Reliability Lead", "evaluation_reliability", "verification_benchmarks"),
    ("experience-lead", "Experience & Interface Lead", "experience_interface", "product_ui"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS org_subsystems (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    department_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mission TEXT NOT NULL,
    paths_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, department_id) REFERENCES org_departments(tenant_id, id),
    FOREIGN KEY (tenant_id, team_id) REFERENCES org_teams(tenant_id, id)
);
CREATE TABLE IF NOT EXISTS org_departments (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mission TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, name)
);
CREATE TABLE IF NOT EXISTS org_teams (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    department_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mission TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, name),
    FOREIGN KEY (tenant_id, department_id) REFERENCES org_departments(tenant_id, id)
);
CREATE TABLE IF NOT EXISTS org_employees (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    career_track TEXT NOT NULL DEFAULT 'individual_contributor',
    level TEXT NOT NULL DEFAULT 'engineer',
    capability_mode TEXT NOT NULL DEFAULT 'engineering',
    system_prompt TEXT NOT NULL DEFAULT '',
    skills_json TEXT NOT NULL DEFAULT '[]',
    primary_ownership_json TEXT NOT NULL DEFAULT '[]',
    secondary_ownership_json TEXT NOT NULL DEFAULT '[]',
    subsystem_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, name),
    FOREIGN KEY (tenant_id, team_id) REFERENCES org_teams(tenant_id, id)
);
CREATE TABLE IF NOT EXISTS org_projects (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'planned',
    lead_employee_id TEXT,
    github_repository TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id),
    UNIQUE (tenant_id, name),
    FOREIGN KEY (tenant_id, lead_employee_id) REFERENCES org_employees(tenant_id, id)
);
CREATE TABLE IF NOT EXISTS org_work_items (
    id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    parent_id TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    item_type TEXT NOT NULL DEFAULT 'task',
    status TEXT NOT NULL DEFAULT 'backlog',
    priority TEXT NOT NULL DEFAULT 'medium',
    subsystem_id TEXT NOT NULL DEFAULT '',
    assignee_employee_id TEXT,
    github_issue_url TEXT NOT NULL DEFAULT '',
    branch_name TEXT NOT NULL DEFAULT '',
    pull_request_url TEXT NOT NULL DEFAULT '',
    github_head_sha TEXT NOT NULL DEFAULT '',
    github_merged INTEGER NOT NULL DEFAULT 0,
    github_reviewed_by TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'not_requested',
    evaluation_status TEXT NOT NULL DEFAULT 'not_run',
    verification_json TEXT NOT NULL DEFAULT '{}',
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, id),
    FOREIGN KEY (tenant_id, project_id) REFERENCES org_projects(tenant_id, id),
    FOREIGN KEY (tenant_id, parent_id) REFERENCES org_work_items(tenant_id, id),
    FOREIGN KEY (tenant_id, assignee_employee_id) REFERENCES org_employees(tenant_id, id)
);
CREATE INDEX IF NOT EXISTS ix_org_teams_department ON org_teams(tenant_id, department_id);
CREATE INDEX IF NOT EXISTS ix_org_employees_team ON org_employees(tenant_id, team_id);
CREATE INDEX IF NOT EXISTS ix_org_work_items_project_status ON org_work_items(tenant_id, project_id, status);
CREATE INDEX IF NOT EXISTS ix_org_work_items_assignee ON org_work_items(tenant_id, assignee_employee_id);
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _database_path() -> Path:
    configured = os.environ.get("SONIC_ORGANIZATION_DB_PATH") or os.environ.get(
        "SONIC_MEMORY_DB_PATH"
    )
    if configured:
        return Path(configured).expanduser().resolve()
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("sqlite"):
        configured_path = db_url.split(":///", 1)[-1] if ":///" in db_url else "sonic_data.db"
        return Path(configured_path).expanduser().resolve()
    return Path("sonic_data.db").resolve()


class OrganizationStore:
    """SQLite-backed organization store, isolated by tenant on every query."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).resolve() if db_path else _database_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(org_work_items)").fetchall()
            }
            migrations = {
                "github_head_sha": "TEXT NOT NULL DEFAULT ''",
                "github_merged": "INTEGER NOT NULL DEFAULT 0",
                "github_reviewed_by": "TEXT NOT NULL DEFAULT ''",
            }
            employee_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(org_employees)").fetchall()
            }
            work_item_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(org_work_items)").fetchall()
            }
            if "capability_mode" not in employee_columns:
                conn.execute(
                    "ALTER TABLE org_employees ADD COLUMN capability_mode "
                    "TEXT NOT NULL DEFAULT 'engineering'"
                )
            if "subsystem_ids_json" not in employee_columns:
                conn.execute(
                    "ALTER TABLE org_employees ADD COLUMN subsystem_ids_json "
                    "TEXT NOT NULL DEFAULT '[]'"
                )
            if "subsystem_id" not in work_item_columns:
                conn.execute(
                    "ALTER TABLE org_work_items ADD COLUMN subsystem_id "
                    "TEXT NOT NULL DEFAULT ''"
                )
            for name, definition in migrations.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE org_work_items ADD COLUMN {name} {definition}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in (
            "skills_json",
            "primary_ownership_json",
            "secondary_ownership_json",
            "subsystem_ids_json",
            "paths_json",
            "verification_json",
        ):
            if key in result:
                target = key.removesuffix("_json")
                result[target] = json.loads(result.pop(key))
        return result

    def _seed(self, conn: sqlite3.Connection, tenant_id: str) -> None:
        for department_id, name, mission in _DEPARTMENTS:
            conn.execute(
                """INSERT OR IGNORE INTO org_departments
                   (id, tenant_id, name, mission) VALUES (?, ?, ?, ?)""",
                (department_id, tenant_id, name, mission),
            )
        for team_id, name, department_id in _TEAMS:
            conn.execute(
                """INSERT OR IGNORE INTO org_teams
                   (id, tenant_id, department_id, name, mission) VALUES (?, ?, ?, ?, ?)""",
                (team_id, tenant_id, department_id, name, f"Owns {name.lower()} outcomes."),
            )
        for subsystem in SUBSYSTEM_OWNERSHIP:
            conn.execute(
                """INSERT OR IGNORE INTO org_subsystems
                   (id, tenant_id, department_id, team_id, name, mission, paths_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    subsystem.subsystem,
                    tenant_id,
                    subsystem.department.value,
                    subsystem.team.value,
                    subsystem.subsystem.replace("_", " ").title(),
                    subsystem.responsibility,
                    self._json(list(subsystem.path_prefixes)),
                ),
            )
        for employee_id, name, department_id, team_id in _SEED_EMPLOYEES:
            conn.execute(
                """INSERT OR IGNORE INTO org_employees
                   (id, tenant_id, team_id, name, role, level, system_prompt,
                    primary_ownership_json, subsystem_ids_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    employee_id,
                    tenant_id,
                    team_id,
                    name,
                    name,
                    "lead",
                    f"Own outcomes for the {department_id.replace('-', ' ')} department. "
                    "Work from verified repository evidence and report limitations honestly.",
                    self._json([department_id]),
                    self._json([subsystem_id for subsystem_id in (
                        "brain" if employee_id == "cognitive-lead" else
                        "computer_use" if employee_id == "computer-lead" else
                        "api" if employee_id == "platform-lead" else
                        "safety" if employee_id == "safety-lead" else
                        "mission_engine" if employee_id == "evaluation-lead" else
                        "dashboard"
                    )]),
                ),
            )

    def snapshot(self, tenant_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._connection() as conn:
            self._seed(conn, tenant_id)
            tables = {
                "departments": "org_departments",
                "teams": "org_teams",
                "subsystems": "org_subsystems",
                "employees": "org_employees",
                "projects": "org_projects",
                "work_items": "org_work_items",
            }
            output: dict[str, list[dict[str, Any]]] = {}
            for name, table in tables.items():
                if name == "employees":
                    rows = conn.execute(
                        """SELECT e.*, t.department_id
                           FROM org_employees e
                           JOIN org_teams t
                             ON t.tenant_id = e.tenant_id AND t.id = e.team_id
                          WHERE e.tenant_id = ? ORDER BY e.created_at, e.id""",
                        (tenant_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        f"SELECT * FROM {table} WHERE tenant_id = ? ORDER BY created_at, id",
                        (tenant_id,),
                    ).fetchall()
                output[name] = [self._decode(row) for row in rows]
            return output

    def create_employee(self, tenant_id: str, values: dict[str, Any]) -> dict[str, Any]:
        employee_id = _new_id("employee")
        with self._connection() as conn:
            self._seed(conn, tenant_id)
            team = conn.execute(
                "SELECT id FROM org_teams WHERE tenant_id = ? AND id = ?",
                (tenant_id, values["team_id"]),
            ).fetchone()
            if team is None:
                raise ValueError("Team does not exist in this tenant")
            subsystem_ids = list(values.get("subsystem_ids") or [])
            for subsystem_id in subsystem_ids:
                subsystem = conn.execute(
                    "SELECT 1 FROM org_subsystems WHERE tenant_id = ? AND id = ? AND team_id = ?",
                    (tenant_id, subsystem_id, values["team_id"]),
                ).fetchone()
                if subsystem is None:
                    raise ValueError(
                        "Every owned subsystem must belong to the employee's team "
                        f"in this tenant: {subsystem_id}"
                    )
            conn.execute(
                """INSERT INTO org_employees
                   (id, tenant_id, team_id, name, role, career_track, level,
                    capability_mode, system_prompt, skills_json, primary_ownership_json,
                    secondary_ownership_json, subsystem_ids_json, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    employee_id, tenant_id, values["team_id"], values["name"],
                    values["role"], values["career_track"], values["level"],
                    values["capability_mode"], values["system_prompt"],
                    self._json(values["skills"]),
                    self._json(values["primary_ownership"]),
                    self._json(values["secondary_ownership"]),
                    self._json(subsystem_ids), values["status"],
                ),
            )
        return next(
            e for e in self.snapshot(tenant_id)["employees"] if e["id"] == employee_id
        )

    def get_employee(self, tenant_id: str, employee_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            self._seed(conn, tenant_id)
            row = conn.execute(
                """SELECT e.*, t.department_id
                   FROM org_employees e
                   JOIN org_teams t
                     ON t.tenant_id = e.tenant_id AND t.id = e.team_id
                  WHERE e.tenant_id = ? AND e.id = ?""",
                (tenant_id, employee_id),
            ).fetchone()
            return self._decode(row) if row else None

    def create_project(self, tenant_id: str, created_by: str, values: dict[str, Any]) -> dict[str, Any]:
        project_id = _new_id("project")
        with self._connection() as conn:
            self._seed(conn, tenant_id)
            lead_id = values.get("lead_employee_id") or None
            if lead_id and conn.execute(
                "SELECT 1 FROM org_employees WHERE tenant_id = ? AND id = ?",
                (tenant_id, lead_id),
            ).fetchone() is None:
                raise ValueError("Project lead does not exist in this tenant")
            conn.execute(
                """INSERT INTO org_projects
                   (id, tenant_id, name, description, status, lead_employee_id,
                    github_repository, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    project_id, tenant_id, values["name"], values["description"],
                    values["status"], lead_id, values["github_repository"], created_by,
                ),
            )
        return next(p for p in self.snapshot(tenant_id)["projects"] if p["id"] == project_id)

    def create_work_item(
        self, tenant_id: str, created_by: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        item_id = _new_id("work")
        with self._connection() as conn:
            self._seed(conn, tenant_id)
            if conn.execute(
                "SELECT 1 FROM org_projects WHERE tenant_id = ? AND id = ?",
                (tenant_id, values["project_id"]),
            ).fetchone() is None:
                raise ValueError("Project does not exist in this tenant")
            assignee = values.get("assignee_employee_id") or None
            if assignee and conn.execute(
                "SELECT 1 FROM org_employees WHERE tenant_id = ? AND id = ? AND status = 'active'",
                (tenant_id, assignee),
            ).fetchone() is None:
                raise ValueError("Assignee does not exist or is inactive in this tenant")
            subsystem_id = values.get("subsystem_id") or ""
            if subsystem_id and conn.execute(
                "SELECT 1 FROM org_subsystems WHERE tenant_id = ? AND id = ?",
                (tenant_id, subsystem_id),
            ).fetchone() is None:
                raise ValueError("Subsystem does not exist in this tenant")
            if assignee and subsystem_id:
                owns = conn.execute(
                    """SELECT 1 FROM org_employees
                       WHERE tenant_id = ? AND id = ?
                       AND EXISTS (SELECT 1 FROM json_each(subsystem_ids_json) WHERE value = ?)""",
                    (tenant_id, assignee, subsystem_id),
                ).fetchone()
                if owns is None:
                    raise ValueError("Assignee does not own the work item's subsystem")
            parent_id = values.get("parent_id") or None
            if parent_id and conn.execute(
                "SELECT 1 FROM org_work_items WHERE tenant_id = ? AND id = ?",
                (tenant_id, parent_id),
            ).fetchone() is None:
                raise ValueError("Parent work item does not exist in this tenant")
            conn.execute(
                """INSERT INTO org_work_items
                   (id, tenant_id, project_id, parent_id, title, description,
                    item_type, priority, subsystem_id, assignee_employee_id,
                    github_issue_url, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item_id, tenant_id, values["project_id"], parent_id,
                    values["title"], values["description"], values["item_type"],
                    values["priority"], subsystem_id, assignee,
                    values["github_issue_url"], created_by,
                ),
            )
        return next(
            item for item in self.snapshot(tenant_id)["work_items"]
            if item["id"] == item_id
        )

    def update_work_item(
        self, tenant_id: str, work_item_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = {
            "status", "assignee_employee_id", "branch_name", "pull_request_url",
            "github_issue_url",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if not updates:
            raise ValueError("No supported work-item fields were provided")
        with self._connection() as conn:
            self._seed(conn, tenant_id)
            existing = conn.execute(
                "SELECT * FROM org_work_items WHERE tenant_id = ? AND id = ?",
                (tenant_id, work_item_id),
            ).fetchone()
            if existing is None:
                raise KeyError("Work item not found")
            if "status" in updates and updates["status"] in {"merged", "released", "evaluated"}:
                raise ValueError("Merge, evaluation, and release states come from verified GitHub events")
            assignee = updates.get("assignee_employee_id")
            if assignee and conn.execute(
                "SELECT 1 FROM org_employees WHERE tenant_id = ? AND id = ? AND status = 'active'",
                (tenant_id, assignee),
            ).fetchone() is None:
                raise ValueError("Assignee does not exist or is inactive in this tenant")
            next_status = updates.get("status", existing["status"])
            if next_status == "released":
                review_status = updates.get("review_status", existing["review_status"])
                evaluation_status = updates.get(
                    "evaluation_status", existing["evaluation_status"]
                )
                if review_status != "approved" or evaluation_status != "passed":
                    raise ValueError(
                        "Release requires approved review and passed evaluation receipts"
                    )
            if "verification_json" in updates:
                updates["verification_json"] = self._json(updates["verification_json"])
            assignments = ", ".join(f"{key} = ?" for key in updates)
            conn.execute(
                f"""UPDATE org_work_items SET {assignments},
                    updated_at = CURRENT_TIMESTAMP WHERE tenant_id = ? AND id = ?""",
                (*updates.values(), tenant_id, work_item_id),
            )
            row = conn.execute(
                "SELECT * FROM org_work_items WHERE tenant_id = ? AND id = ?",
                (tenant_id, work_item_id),
            ).fetchone()
            return self._decode(row)

    @staticmethod
    def _repository_key(value: str) -> tuple[str, str] | None:
        parsed = urlparse(value.strip())
        path = parsed.path.strip("/").removesuffix(".git").lower()
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not path:
            return None
        return parsed.hostname.lower(), path

    def record_github_event(self, event_name: str, payload: dict[str, Any]) -> int:
        """Apply authenticated GitHub PR, review, and aggregate check-suite receipts."""
        repository = payload.get("repository") or {}
        repo_key = self._repository_key(
            str(repository.get("html_url") or repository.get("url") or "")
        )
        if repo_key is None:
            return 0

        event = event_name.lower()
        action = str(payload.get("action") or "")
        pull_request = payload.get("pull_request") or {}
        pull_request_url = str(pull_request.get("html_url") or "")
        head = pull_request.get("head") or {}
        branch = str(head.get("ref") or "")
        head_sha = str(head.get("sha") or "")
        updates: list[tuple[str, tuple[Any, ...]]] = []

        if event == "pull_request":
            if not pull_request_url:
                return 0
            merged = int(action == "closed" and bool(pull_request.get("merged")))
            if action in {"opened", "reopened", "synchronize", "ready_for_review"}:
                updates = [
                    (
                        """pull_request_url = ?, branch_name = ?, github_head_sha = ?,
                           github_merged = 0, github_reviewed_by = '',
                           review_status = 'pending', evaluation_status = 'not_run',
                           verification_json = '{}', status = 'in_review'""",
                        (pull_request_url, branch, head_sha),
                    )
                ]
            elif action == "closed":
                updates = [
                    (
                        """pull_request_url = ?, branch_name = ?, github_head_sha = ?,
                           github_merged = ?, status = CASE WHEN ? = 1 THEN 'merged' ELSE 'blocked' END""",
                        (pull_request_url, branch, head_sha, merged, merged),
                    )
                ]
            else:
                return 0
            match_field = "pull_request_url = ? OR (branch_name = ? AND github_head_sha IN ('', ?))"
            match_values: tuple[Any, ...] = (pull_request_url, branch, head_sha)
        elif event == "pull_request_review" and action == "submitted":
            review = payload.get("review") or {}
            review_sha = str(review.get("commit_id") or "")
            state = str(review.get("state") or "").lower()
            review_status = {
                "approved": "approved",
                "changes_requested": "changes_requested",
            }.get(state, "pending")
            if not pull_request_url or not review_sha:
                return 0
            updates = [
                (
                    "review_status = ?, github_reviewed_by = ?",
                    (
                        review_status,
                        str(((payload.get("review") or {}).get("user") or {}).get("login") or ""),
                    ),
                )
            ]
            match_field = "pull_request_url = ? AND github_head_sha = ?"
            match_values = (pull_request_url, review_sha)
        elif event == "check_suite":
            suite = payload.get("check_suite") or {}
            if action != "completed" or str(suite.get("status") or "") != "completed":
                return 0
            suite_sha = str(suite.get("head_sha") or "")
            suite_branch = str(suite.get("head_branch") or "")
            conclusion = str(suite.get("conclusion") or "").lower()
            suite_url = str(suite.get("html_url") or "")
            suite_id = str(suite.get("id") or "")
            if not suite_sha or not suite_branch:
                return 0
            evaluation_status = "passed" if conclusion == "success" else "failed"
            receipt = self._json(
                {
                    "source": "github_check_suite",
                    "check_suite_id": suite_id,
                    "url": suite_url,
                    "head_sha": suite_sha,
                    "conclusion": conclusion,
                    "completed_at": suite.get("updated_at") or suite.get("completed_at") or "",
                }
            )
            updates = [
                (
                    "evaluation_status = ?, verification_json = ?",
                    (evaluation_status, receipt),
                )
            ]
            match_field = "branch_name = ? AND github_head_sha = ?"
            match_values = (suite_branch, suite_sha)
        else:
            return 0

        total_updated = 0
        with self._connection() as conn:
            projects = conn.execute(
                "SELECT tenant_id, id, github_repository FROM org_projects"
            ).fetchall()
            matching_projects = [
                (row["tenant_id"], row["id"])
                for row in projects
                if self._repository_key(row["github_repository"]) == repo_key
            ]
            for tenant_id, project_id in matching_projects:
                for assignment, update_values in updates:
                    cursor = conn.execute(
                        f"""UPDATE org_work_items SET {assignment},
                            updated_at = CURRENT_TIMESTAMP
                            WHERE tenant_id = ? AND project_id = ? AND ({match_field})""",
                        (*update_values, tenant_id, project_id, *match_values),
                    )
                    total_updated += cursor.rowcount
                conn.execute(
                    """UPDATE org_work_items
                       SET status = CASE
                           WHEN github_merged = 1 AND review_status = 'approved'
                                AND evaluation_status = 'passed' THEN 'released'
                           WHEN github_merged = 1 THEN 'merged'
                           ELSE status END,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE tenant_id = ? AND project_id = ?""",
                    (tenant_id, project_id),
                )
        return total_updated


_store: OrganizationStore | None = None
_store_path: Path | None = None


def get_organization_store() -> OrganizationStore:
    global _store, _store_path
    path = _database_path()
    if _store is None or _store_path != path:
        _store = OrganizationStore(path)
        _store_path = path
    return _store


def reset_organization_store() -> None:
    global _store, _store_path
    _store = None
    _store_path = None
