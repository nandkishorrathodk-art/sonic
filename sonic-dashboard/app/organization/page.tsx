"use client";

import { useEffect, useMemo, useState } from "react";
import { BriefcaseBusiness, Building2, CheckCircle2, Users } from "lucide-react";
import { api } from "../../lib/api";

type Organization = {
  departments: string[];
  teams: { id: string; department_id: string; mission: string }[];
  subsystems: { id: string; department_id: string; team_id: string; path_prefixes: string[]; responsibility: string }[];
  employees: { id: string; name: string; role: string; department_id: string; team_id: string; subsystem_ids: string[]; read_only: boolean; status: string }[];
  projects: { id: string; name: string; description: string; lead_employee_id?: string | null; status: string }[];
  work_items: { id: string; project_id: string; title: string; subsystem_id: string; department_id: string; team_id: string; assignee_employee_id: string; github_repo: string; github_issue_number?: number | null; github_pr_number?: number | null; development_branch: string; status: string }[];
};

function label(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

const emptyEmployee = {
  name: "",
  role: "AI Agent Engineer",
  department_id: "cognitive_systems",
  team_id: "reasoning_orchestration",
  subsystem_ids: ["agents"],
  read_only: false,
  employee_type: "ai",
  prompt: "",
};

export default function OrganizationPage() {
  const [org, setOrg] = useState<Organization | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [employee, setEmployee] = useState(emptyEmployee);
  const [project, setProject] = useState({ name: "", description: "", lead_employee_id: "" });
  const [workItem, setWorkItem] = useState({
    project_id: "", title: "", description: "", subsystem_id: "", employee_id: "",
    github_repo: "", github_issue_number: "", github_pr_number: "", development_branch: "",
  });

  async function refresh() {
    try {
      setOrg(await api.getOrganization());
      setError("");
    } catch (err: any) {
      setError(err?.message || "Failed to load organization.");
    }
  }

  useEffect(() => { void refresh(); }, []);

  const teams = useMemo(
    () => org?.teams.filter((team) => team.department_id === employee.department_id) || [],
    [org, employee.department_id],
  );
  const employeeSubsystems = useMemo(
    () => org?.subsystems.filter((item) => item.department_id === employee.department_id && item.team_id === employee.team_id) || [],
    [org, employee.department_id, employee.team_id],
  );
  const assignableEmployees = useMemo(
    () => org?.employees.filter((item) => item.subsystem_ids.includes(workItem.subsystem_id)) || [],
    [org, workItem.subsystem_id],
  );

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await action();
      await refresh();
    } catch (err: any) {
      setError(err?.message || "Organization operation failed.");
    } finally {
      setBusy(false);
    }
  }

  if (!org) return <div className="p-6 text-sm text-muted">{error || "Loading organization..."}</div>;

  return (
    <div className="min-h-screen bg-ink-950 text-slate-200 p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white">SONIC Organization</h1>
        <p className="text-sm text-muted mt-1">Department → Team → Employee → Subsystem → Project → Work Item → Evaluation.</p>
      </div>

      {error && <div className="rounded border border-danger/40 bg-danger/10 p-3 text-sm text-danger">{error}</div>}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <section className="panel">
          <div className="panel-title"><Building2 className="w-4 h-4" /> Departments</div>
          <div className="mt-3 space-y-2">
            {org.departments.map((id) => (
              <div key={id} className="card">
                <div className="text-white">{label(id)}</div>
                <div className="text-[11px] text-muted">{org.teams.filter((team) => team.department_id === id).map((team) => label(team.id)).join(" · ")}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title"><Users className="w-4 h-4" /> AI Employees</div>
          <div className="mt-3 space-y-2">
            {org.employees.map((item) => (
              <div key={item.id} className="card">
                <div className="text-white font-medium">{item.name}</div>
                <div className="text-xs text-muted">{item.role}</div>
                <div className="text-[11px] text-secondary-300 mt-1">{label(item.department_id)} / {label(item.team_id)}</div>
                <div className="text-[11px] text-muted-dim mt-1">{item.subsystem_ids.join(", ")}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-title"><BriefcaseBusiness className="w-4 h-4" /> Projects</div>
          <div className="mt-3 space-y-2">
            {org.projects.map((item) => (
              <div key={item.id} className="card">
                <div className="text-white font-medium">{item.name}</div>
                <div className="text-xs text-muted">{item.description || "No description"}</div>
                <div className="text-[11px] text-muted-dim mt-1">{item.status}</div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <section className="panel space-y-3">
          <h2 className="font-semibold text-white">Create AI Employee</h2>
          <input className="sonic-input" placeholder="Name" value={employee.name} onChange={(e) => setEmployee({ ...employee, name: e.target.value })} />
          <input className="sonic-input" placeholder="Role" value={employee.role} onChange={(e) => setEmployee({ ...employee, role: e.target.value })} />
          <select className="sonic-input" value={employee.department_id} onChange={(e) => {
            const department_id = e.target.value;
            const next = org.teams.find((team) => team.department_id === department_id);
            setEmployee({ ...employee, department_id, team_id: next?.id || "", subsystem_ids: [] });
          }}>
            {org.departments.map((id) => <option key={id} value={id}>{label(id)}</option>)}
          </select>
          <select className="sonic-input" value={employee.team_id} onChange={(e) => {
            const team_id = e.target.value;
            const next = org.subsystems.find((item) => item.department_id === employee.department_id && item.team_id === team_id);
            setEmployee({ ...employee, team_id, subsystem_ids: next ? [next.id] : [] });
          }}>
            {teams.map((team) => <option key={team.id} value={team.id}>{label(team.id)}</option>)}
          </select>
          <select className="sonic-input" value={employee.subsystem_ids[0] || ""} onChange={(e) => setEmployee({ ...employee, subsystem_ids: [e.target.value] })}>
            {employeeSubsystems.map((item) => <option key={item.id} value={item.id}>{item.id}</option>)}
          </select>
          <label className="text-xs text-muted flex items-center gap-2">
            <input type="checkbox" checked={employee.read_only} onChange={(e) => setEmployee({ ...employee, read_only: e.target.checked })} /> Read-only employee
          </label>
          <button disabled={busy} className="sonic-button" onClick={() => run(async () => {
            await api.createEmployee(employee);
            setEmployee({ ...emptyEmployee });
          })}>Create Employee</button>
        </section>

        <section className="panel space-y-3">
          <h2 className="font-semibold text-white">Create Project</h2>
          <input className="sonic-input" placeholder="Project name" value={project.name} onChange={(e) => setProject({ ...project, name: e.target.value })} />
          <textarea className="sonic-input min-h-24" placeholder="Description" value={project.description} onChange={(e) => setProject({ ...project, description: e.target.value })} />
          <select className="sonic-input" value={project.lead_employee_id} onChange={(e) => setProject({ ...project, lead_employee_id: e.target.value })}>
            <option value="">No lead</option>
            {org.employees.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          <button disabled={busy} className="sonic-button" onClick={() => run(async () => {
            await api.createOrganizationProject({ ...project, lead_employee_id: project.lead_employee_id || null });
            setProject({ name: "", description: "", lead_employee_id: "" });
          })}>Create Project</button>
        </section>

        <section className="panel space-y-3">
          <h2 className="font-semibold text-white">Assign Work Item</h2>
          <select className="sonic-input" value={workItem.project_id} onChange={(e) => setWorkItem({ ...workItem, project_id: e.target.value })}>
            <option value="">Project</option>
            {org.projects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          <input className="sonic-input" placeholder="Work item title" value={workItem.title} onChange={(e) => setWorkItem({ ...workItem, title: e.target.value })} />
          <select className="sonic-input" value={workItem.subsystem_id} onChange={(e) => setWorkItem({ ...workItem, subsystem_id: e.target.value, employee_id: "" })}>
            <option value="">Subsystem</option>
            {org.subsystems.map((item) => <option key={item.id} value={item.id}>{item.id} — {label(item.team_id)}</option>)}
          </select>
          <select className="sonic-input" value={workItem.employee_id} onChange={(e) => setWorkItem({ ...workItem, employee_id: e.target.value })}>
            <option value="">Employee</option>
            {assignableEmployees.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          <input className="sonic-input" placeholder="GitHub repo (owner/repo)" value={workItem.github_repo} onChange={(e) => setWorkItem({ ...workItem, github_repo: e.target.value })} />
          <input className="sonic-input" placeholder="Development branch" value={workItem.development_branch} onChange={(e) => setWorkItem({ ...workItem, development_branch: e.target.value })} />
          <button disabled={busy} className="sonic-button" onClick={() => run(async () => {
            await api.createWorkItem({
              ...workItem,
              github_issue_number: workItem.github_issue_number ? Number(workItem.github_issue_number) : null,
              github_pr_number: workItem.github_pr_number ? Number(workItem.github_pr_number) : null,
            });
            setWorkItem({ project_id: "", title: "", description: "", subsystem_id: "", employee_id: "", github_repo: "", github_issue_number: "", github_pr_number: "", development_branch: "" });
          })}>Assign Work Item</button>
        </section>
      </div>

      <section className="panel">
        <div className="panel-title"><CheckCircle2 className="w-4 h-4" /> Work Items</div>
        <div className="mt-3 space-y-2">
          {org.work_items.map((item) => (
            <div key={item.id} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-ink-800 bg-ink-950 p-3">
              <div>
                <div className="text-white">{item.title}</div>
                <div className="text-[11px] text-muted">{item.department_id} / {item.team_id} / {item.subsystem_id}</div>
                <div className="text-[11px] text-muted-dim">
                  {item.github_repo || "No GitHub reference"}
                  {item.github_pr_number ? " PR #" + item.github_pr_number : ""}
                </div>
              </div>
              <select className="sonic-input w-40" value={item.status} onChange={(e) => run(() => api.updateWorkItemStatus(item.id, e.target.value))}>
                {["assigned", "in_progress", "review", "evaluating", "blocked", "released"].map((status) => <option key={status} value={status}>{status}</option>)}
              </select>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
