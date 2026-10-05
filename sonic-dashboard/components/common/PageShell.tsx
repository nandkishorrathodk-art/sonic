import React from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";

interface PageHeaderProps {
  icon: React.ReactNode;
  title: string;
  badge?: string;
  subtitle?: string;
  actions?: React.ReactNode;
  accent?: "primary" | "secondary" | "accent" | "success";
}

export function PageHeader({ icon, title, badge, subtitle, actions, accent = "primary" }: PageHeaderProps) {
  return (
    <div className="sonic-page-header flex flex-col gap-4 md:flex-row md:items-end md:justify-between animate-fade-in-up" data-accent={accent}>
      <div className="flex items-center gap-3.5 min-w-0">
        <div className="sonic-page-header-icon grid place-items-center w-10 h-10 rounded-xl shrink-0">
          {icon}
        </div>
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-xl md:text-2xl font-semibold text-white tracking-tight truncate">{title}</h1>
            {badge && (
              <span className="chip bg-ink-850 text-muted-bright border border-ink-700">
                {badge}
              </span>
            )}
          </div>
          {subtitle && (
            <p className="text-xs text-muted mt-1.5 truncate">{subtitle}</p>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <Link
          href="/workstation"
          className="btn-ghost !px-3 !py-2 text-xs text-muted hover:text-white flex items-center gap-1.5 border border-ink-700/60 rounded-lg hover:bg-ink-800 transition"
          title="Return to Workstation"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Workstation</span>
        </Link>
        {actions}
      </div>
    </div>
  );
}

interface PageShellProps {
  children: React.ReactNode;
  maxWidth?: string;
}

export function PageShell({ children, maxWidth = "max-w-7xl" }: PageShellProps) {
  return (
    <main className={`sonic-page-shell min-h-screen bg-ink-950 ${maxWidth} mx-auto p-5 sm:p-6 space-y-6`}>
      {children}
    </main>
  );
}

interface StateBlockProps {
  error?: string | null;
  loading?: boolean;
  empty?: boolean;
  emptyIcon?: React.ReactNode;
  emptyTitle?: string;
  emptyText?: string;
  spinnerColor?: string;
  loadingText?: string;
  children?: React.ReactNode;
}

export function StateBlock({
  error,
  loading,
  empty,
  emptyIcon,
  emptyTitle,
  emptyText,
  spinnerColor = "text-primary-400",
  loadingText = "Loading…",
  children,
}: StateBlockProps) {
  if (error) {
    return (
      <div className="p-4 rounded-xl bg-danger/10 border border-danger/30 text-danger text-xs font-mono">
        <strong className="mr-1">Backend Error:</strong> {error}
      </div>
    );
  }
  if (loading) {
    return (
      <div className="py-20 text-center space-y-3 animate-fade-in-up">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-ink-850 border border-ink-700">
          <span className={`block w-6 h-6 rounded-full border-2 border-ink-600 border-t-transparent ${spinnerColor} animate-devin-spin`} />
        </div>
        <span className="text-xs text-muted font-mono">{loadingText}</span>
      </div>
    );
  }
  if (empty) {
    return (
      <div className="glass-card rounded-2xl p-12 text-center space-y-3 animate-fade-in-up">
        <div className="grid place-items-center w-16 h-16 mx-auto rounded-2xl bg-ink-800 border border-ink-700 text-muted-dim">
          {emptyIcon}
        </div>
        {emptyTitle && <h3 className="text-sm font-bold text-white">{emptyTitle}</h3>}
        {emptyText && (
          <p className="text-xs text-muted max-w-md mx-auto">{emptyText}</p>
        )}
      </div>
    );
  }
  return <>{children}</>;
}
