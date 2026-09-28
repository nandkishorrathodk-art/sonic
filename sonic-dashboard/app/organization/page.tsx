"use client";

import { type FormEvent, useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  BriefcaseBusiness,
  Building2,
  Plus,
  RefreshCw,
  UsersRound,
} from "lucide-react";
import { api } from "../../lib/api";
import { PageHeader, PageShell, StateBlock } from "../../components/common/PageShell";

interface Department {
  id: string;
  name: string;
  mission: string;
}

interface Team {
  id: string;
  department_id: string;
  name: string;
  mission: string;
}

interface Employee {
  id: string;
  team_id: string;
  name: string;
  role: string;
  level: string;
  status: string;
  capability_mode: "read_only" | "engineering";
  primary_ownership: string[];
  secondary_ownership: string[];
}

interface Project {
  id: string;
  name: string;
  description: string;
  status: string;
  lead_employee_id: string | null;
  github_repository: string;
}

interface WorkItem {
  id: string;
  project_id: string;
  title: string;
  description: string;
  item_type: string;
  status: string;
  priority: string;
  assignee_employee_id: string | null;
  github_issue_url: string;
  branch_name: string;
  pull_request_url: string;
  review_status: string;
  evaluation_status: string;
}

interface OrganizationState {
  departments: Department[];
  teams: Team[];
  employees: Employee[];
  projects: Project[];
  work_items: WorkItem[];
}

const emptyState: OrganizationState = {
  departments: [],
  teams: [],
  employees: [],
  projects: [],
  work_items: [],
};

const inputClass =
  "w-full rounded-lg border border-ink-700 bg-ink-900 px-3 py-2 text-xs text-white outline-none focus:border-primary-500";

export default function OrganizationPage() {
  const [organization, setOrganization] = useState<OrganizationState>(emptyState);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [employeeName, setEmployeeName] = useState("");
  const [employeeRole, setEmployeeRole] = useState("");
  const [employeeTeam, setEmployeeTeam] = useState("");
  const [employeeOwnership, setEmployeeOwnership] = useState("");
  const [employeeMode, setEmployeeMode] = useState<"read_only" | "engineering">("engineering");
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [projectRepository, setProjectRepository] = useState("");
  const [projectLead, setProjectLead] = useState("");
  const [workTitle, setWorkTitle] = useState("");
  const [workProject, setWorkProject] = useState("");
  const [workAssignee, setWorkAssignee] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = (await api.getOrganization()) as OrganizationState;
      setOrganization(data);
      setEmployeeTeam((current) => current || data.teams[0]?.id || "");
      setWorkProject((current) => current || data.projects[0]?.id || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the organization.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function submit(
    event: FormEvent,
    operation: () => Promise<unknown>,
  ) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await operation();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "The change could not be saved.");
    } finally {
      setSaving(false);
    }
  }

  const employeeNameById = new Map(
    organization.employees.map((employee) => [employee.id, employee.name]),
  );

  return (
    <PageShell>
      <PageHeader
        accent="primary"
        icon={<UsersRound className="h-6 w-6 text-primary-400" />}
        title="SONIC ORGANIZATION"
        badge={`${organization.employees.length} AI roles · ${organization.projects.length} projects`}
        subtitle="Persistent departments, role profiles, ownership, and assigned work"
        actions={
          <button
            onClick={() => void refresh()}
            disabled={loading}
            className="btn-ghost inline-flex items-center gap-2 text-xs font-mono"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            REFRESH
          </button>
        }
      />

      {error && (
        <div className="flex items-center gap-2 rounded-xl border border-danger/30 bg-danger/10 p-4 text-xs text-danger">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      <StateBlock
        loading={loading}
        loadingText="Loading tenant organization…"
        empty={false}
        emptyIcon={<Building2 className="h-6 w-6" />}
        emptyTitle="Organization unavailable"
        emptyText="The organization API did not return its structure."
        children={
          <div className="space-y-8">
            <section className="space-y-3">
              <div>
                <h2 className="text-sm font-bold text-white">Departments & teams</h2>
                <p className="mt-1 text-xs text-muted">
                  Cross-functional research stays separate from permanent delivery ownership.
                </p>
              </div>
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                {organization.departments.map((department) => (
                  <article key={department.id} className="glass-card rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-white">{department.name}</h3>
                    <p className="mt-1 text-xs text-muted">{department.mission}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {organization.teams
                        .filter((team) => team.department_id === department.id)
                        .map((team) => (
                          <span
                            key={team.id}
                            className="rounded-md border border-ink-700 bg-ink-900 px-2 py-1 text-[10px] text-secondary-300"
                            title={team.mission}
                          >
                            {team.name}
                          </span>
                        ))}
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="space-y-3">
              <div>
                <h2 className="text-sm font-bold text-white">AI employee profiles</h2>
                <p className="mt-1 text-xs text-muted">
                  Role profiles organize work; they are not human accounts or proof of a running agent.
                </p>
              </div>
              <form
                className="glass-card grid gap-2 rounded-xl p-4 md:grid-cols-[1fr_1fr_1fr_1fr_auto]"
                onSubmit={(event) =>
                  void submit(event, () =>
                    api.createOrganizationEmployee({
                      name: employeeName,
                      role: employeeRole,
                      team_id: employeeTeam,
                      primary_ownership: employeeOwnership
                        .split(",")
                        .map((item) => item.trim())
                        .filter(Boolean),
                      capability_mode: employeeMode,
                    }),
                  )
                }
              >
                <input
                  className={inputClass}
                  value={employeeName}
                  onChange={(event) => setEmployeeName(event.target.value)}
                  placeholder="Role profile name"
                  required
                  maxLength={128}
                />
                <input
                  className={inputClass}
                  value={employeeRole}
                  onChange={(event) => setEmployeeRole(event.target.value)}
                  placeholder="Role / responsibility"
                  required
                  maxLength={128}
                />
                <input
                  className={inputClass}
                  value={employeeOwnership}
                  onChange={(event) => setEmployeeOwnership(event.target.value)}
                  placeholder="Subsystem ownership (comma separated)"
                  maxLength={500}
                />
                <select
                  className={inputClass}
                  value={employeeTeam}
                  onChange={(event) => setEmployeeTeam(event.target.value)}
                  required
                >
                  {organization.teams.map((team) => (
                    <option key={team.id} value={team.id}>{team.name}</option>
                  ))}
                </select>
                <select
                  className={inputClass}
                  value={employeeMode}
                  onChange={(event) => setEmployeeMode(event.target.value as "read_only" | "engineering")}
                  aria-label="Employee capability mode"
                >
                  <option value="engineering">Engineering access</option>
                  <option value="read_only">Read only</option>
                </select>
                <button className="btn-primary inline-flex items-center justify-center gap-1 px-3 text-xs" disabled={saving}>
                  <Plus className="h-3 w-3" /> ADD ROLE
                </button>
              </form>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                {organization.employees.map((employee) => {
                  const team = organization.teams.find((item) => item.id === employee.team_id);
                  const department = organization.departments.find(
                    (item) => item.id === team?.department_id,
                  );
                  return (
                    <article key={employee.id} className="glass-card rounded-xl p-4">
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <h3 className="text-sm font-semibold text-white">{employee.name}</h3>
                          <p className="text-xs text-muted">{employee.role} · {employee.level}</p>
                        </div>
                        <span className="rounded border border-success/30 px-2 py-0.5 text-[10px] uppercase text-success">
                          {employee.status}
                        </span>
                      </div>
                      <p className="mt-2 text-[11px] text-muted-dim">{department?.name} / {team?.name}</p>
                      <p className="mt-2 text-[11px] text-muted">
                        {employee.capability_mode.replaceAll("_", " ")} · Owns: {[...employee.primary_ownership, ...employee.secondary_ownership].join(", ") || "No subsystem ownership assigned"}
                      </p>
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="grid gap-5 xl:grid-cols-2">
              <div className="space-y-3">
                <div>
                  <h2 className="text-sm font-bold text-white">Projects</h2>
                  <p className="mt-1 text-xs text-muted">Each project has an accountable lead and repository reference.</p>
                </div>
                <form
                  className="glass-card space-y-2 rounded-xl p-4"
                  onSubmit={(event) =>
                    void submit(event, () =>
                      api.createOrganizationProject({
                        name: projectName,
                        description: projectDescription,
                        lead_employee_id: projectLead || null,
                        github_repository: projectRepository,
                      }),
                    )
                  }
                >
                  <input className={inputClass} value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="Project name" required maxLength={160} />
                  <textarea className={inputClass} value={projectDescription} onChange={(event) => setProjectDescription(event.target.value)} placeholder="Objective and acceptance outcome" rows={2} />
                  <input className={inputClass} value={projectRepository} onChange={(event) => setProjectRepository(event.target.value)} placeholder="Repository URL (optional)" maxLength={500} />
                  <select className={inputClass} value={projectLead} onChange={(event) => setProjectLead(event.target.value)}>
                    <option value="">No lead assigned</option>
                    {organization.employees.map((employee) => <option key={employee.id} value={employee.id}>{employee.name}</option>)}
                  </select>
                  <button className="btn-primary inline-flex items-center gap-1 px-3 text-xs" disabled={saving}><Plus className="h-3 w-3" /> CREATE PROJECT</button>
                </form>
                <div className="space-y-2">
                  {organization.projects.map((project) => (
                    <article key={project.id} className="glass-card rounded-xl p-3">
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="text-xs font-semibold text-white">{project.name}</h3>
                        <span className="text-[10px] uppercase text-secondary-300">{project.status}</span>
                      </div>
                      <p className="mt-1 text-[11px] text-muted">{project.description || "No project description."}</p>
                      <p className="mt-1 text-[10px] text-muted-dim">Lead: {project.lead_employee_id ? employeeNameById.get(project.lead_employee_id) : "Unassigned"} · Repository: {project.github_repository || "Not linked"}</p>
                    </article>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <h2 className="text-sm font-bold text-white">Work items & delivery gates</h2>
                  <p className="mt-1 text-xs text-muted">Track assignment, issue/PR references, review, and evaluation receipts.</p>
                </div>
                <form
                  className="glass-card space-y-2 rounded-xl p-4"
                  onSubmit={(event) =>
                    void submit(event, () =>
                      api.createOrganizationWorkItem({
                        project_id: workProject,
                        title: workTitle,
                        assignee_employee_id: workAssignee || null,
                      }),
                    )
                  }
                >
                  <input className={inputClass} value={workTitle} onChange={(event) => setWorkTitle(event.target.value)} placeholder="Task / issue title" required maxLength={255} />
                  <select className={inputClass} value={workProject} onChange={(event) => setWorkProject(event.target.value)} required>
                    <option value="">Choose project</option>
                    {organization.projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
                  </select>
                  <select className={inputClass} value={workAssignee} onChange={(event) => setWorkAssignee(event.target.value)}>
                    <option value="">Unassigned</option>
                    {organization.employees.map((employee) => <option key={employee.id} value={employee.id}>{employee.name}</option>)}
                  </select>
                  <button className="btn-primary inline-flex items-center gap-1 px-3 text-xs" disabled={saving || organization.projects.length === 0}><Plus className="h-3 w-3" /> CREATE WORK ITEM</button>
                </form>
                <div className="space-y-2">
                  {organization.work_items.map((item) => (
                    <form
                      key={item.id}
                      className="glass-card space-y-2 rounded-xl p-3"
                      onSubmit={(event) => {
                        const form = new FormData(event.currentTarget);
                        void submit(event, () =>
                          api.updateOrganizationWorkItem(item.id, {
                            status: form.get("status"),
                            assignee_employee_id: form.get("assignee_employee_id") || null,
                            github_issue_url: form.get("github_issue_url") || "",
                            branch_name: form.get("branch_name") || "",
                            pull_request_url: form.get("pull_request_url") || "",
                          }),
                        );
                      }}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="text-xs font-semibold text-white">{item.title}</h3>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-dim">
                        <span>{item.priority} priority</span>
                        <span>Owner: {item.assignee_employee_id ? employeeNameById.get(item.assignee_employee_id) : "Unassigned"}</span>
                        <span>Review: {item.review_status}</span>
                        <span>Evaluation: {item.evaluation_status}</span>
                      </div>
                      <select className={inputClass} name="assignee_employee_id" defaultValue={item.assignee_employee_id || ""}>
                        <option value="">Unassigned</option>
                        {organization.employees.map((employee) => <option key={employee.id} value={employee.id}>{employee.name}</option>)}
                      </select>
                      <select className={inputClass} name="status" defaultValue={item.status}>
                        {["backlog", "ready", "in_progress", "blocked", "in_review", "evaluated"].map((status) => (
                          <option key={status} value={status}>{status.replaceAll("_", " ")}</option>
                        ))}
                      </select>
                      <div className="grid gap-2 md:grid-cols-3">
                        <input className={inputClass} name="github_issue_url" defaultValue={item.github_issue_url} placeholder="Issue URL" />
                        <input className={inputClass} name="branch_name" defaultValue={item.branch_name} placeholder="Development branch reference" />
                        <input className={inputClass} name="pull_request_url" defaultValue={item.pull_request_url} placeholder="PR URL" />
                      </div>
                      <button className="btn-ghost px-3 py-1 text-[10px]" disabled={saving}>SAVE ASSIGNMENT / REFERENCES</button>
                    </form>
                  ))}
                  {organization.work_items.length === 0 && (
                    <div className="glass-card rounded-xl p-4 text-xs text-muted">No work items recorded for this tenant.</div>
                  )}
                </div>
              </div>
            </section>

            <div className="flex items-center gap-2 border-t border-ink-800 pt-4 text-[10px] text-muted-dim">
              <BriefcaseBusiness className="h-3 w-3" />
              Issue/PR links and work status are operator-maintained records. This registry does not create branches, execute CI, approve reviews, merge, or release changes.
            </div>
          </div>
        }
      />
    </PageShell>
  );
}
