"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Bot,
  Check,
  ChevronRight,
  CircleDot,
  Code2,
  Command,
  Eye,
  FileCheck2,
  Fingerprint,
  Layers3,
  LockKeyhole,
  Menu,
  Monitor,
  Network,
  Orbit,
  Play,
  Radio,
  ShieldCheck,
  Sparkles,
  Terminal,
  Workflow,
  X,
} from "lucide-react";
import { API_BASE } from "../../lib/api";
import { BrandMark } from "../../components/common/BrandMark";

const HeroScene = dynamic(
  () => import("../../components/landing/HeroScene").then((module) => module.HeroScene),
  {
    ssr: false,
    loading: () => (
      <div className="sonic-hero-scene" aria-hidden="true">
        <div className="sonic-hero-scene-fallback" />
      </div>
    ),
  },
);

const WORKFLOW = [
  {
    number: "01",
    icon: Eye,
    title: "See the whole task",
    description: "SONIC observes the workspace and the live desktop before choosing what to do next.",
    detail: "Observe · establish state",
  },
  {
    number: "02",
    icon: Workflow,
    title: "Put the right agents to work",
    description: "A coordinating agent can divide a goal into focused work across GUI and terminal.",
    detail: "Delegate · work in parallel",
  },
  {
    number: "03",
    icon: FileCheck2,
    title: "Verify before reporting",
    description: "Actions and outcomes are checked against their results—not just declared complete.",
    detail: "Verify · show the evidence",
  },
];

const CAPABILITIES = [
  {
    icon: Monitor,
    label: "COMPUTER",
    title: "A real desktop, not a mock",
    description: "Use a graphical Linux workstation with a visible browser, windows, and human takeover.",
    accent: "mint",
  },
  {
    icon: Terminal,
    label: "ENGINEERING",
    title: "GUI and terminal, together",
    description: "Move between visual computer work and focused command-line execution as the task needs.",
    accent: "violet",
  },
  {
    icon: Network,
    label: "ORCHESTRATION",
    title: "Parallel work with clear roles",
    description: "Delegate distinct responsibilities, follow worker progress, and bring findings back together.",
    accent: "coral",
  },
];

const PRINCIPLES = [
  { icon: ShieldCheck, title: "Safety before speed", text: "Actions stay within an explicit policy boundary." },
  { icon: Fingerprint, title: "Scoped by design", text: "Authentication and tenant context belong to the operator." },
  { icon: Check, title: "Evidence over claims", text: "A completed task should have observable results." },
];

export default function LandingPage() {
  const [systemStatus, setSystemStatus] = useState<"checking" | "online" | "unavailable">("checking");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API_BASE}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Health check returned ${response.status}`);
        return response.json();
      })
      .then((data) => {
        const isAvailable =
          data?.overall_status === "healthy" ||
          data?.overall_status === "degraded" ||
          data?.components?.control_plane_api?.status === "healthy";
        setSystemStatus(isAvailable ? "online" : "unavailable");
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          console.warn("SONIC landing page could not verify API health.", error);
          setSystemStatus("unavailable");
        }
      });
    return () => controller.abort();
  }, []);

  const statusLabel = {
    checking: "CHECKING SYSTEM",
    online: "CONTROL PLANE ONLINE",
    unavailable: "STATUS UNVERIFIED",
  }[systemStatus];

  return (
    <main className="sonic-landing min-h-screen overflow-hidden bg-[#080b10] text-white">
      <div className="sonic-page-glow" aria-hidden="true" />
      <header className="sonic-nav">
        <div className="mx-auto flex h-[76px] max-w-[1320px] items-center justify-between px-5 sm:px-8">
          <BrandMark withWordmark href="/" />

          <nav className="hidden items-center gap-8 md:flex" aria-label="Main navigation">
            <a href="#capabilities" className="sonic-nav-link">Capabilities</a>
            <a href="#how-it-works" className="sonic-nav-link">How it works</a>
            <Link href="/about" className="sonic-nav-link">About SONIC</Link>
          </nav>

          <div className="hidden items-center gap-3 md:flex">
            <Link href="/login" className="sonic-login-link">Sign in</Link>
            <Link href="/login" className="sonic-button sonic-button-small">
              Open SONIC <ArrowUpRight size={15} />
            </Link>
          </div>

          <button
            type="button"
            className="sonic-mobile-toggle md:hidden"
            aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={mobileMenuOpen}
            onClick={() => setMobileMenuOpen((open) => !open)}
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
        {mobileMenuOpen && (
          <nav className="sonic-mobile-menu md:hidden" aria-label="Mobile navigation">
            <a href="#capabilities" onClick={() => setMobileMenuOpen(false)}>Capabilities</a>
            <a href="#how-it-works" onClick={() => setMobileMenuOpen(false)}>How it works</a>
            <Link href="/about" onClick={() => setMobileMenuOpen(false)}>About SONIC</Link>
            <Link href="/login" className="sonic-mobile-cta" onClick={() => setMobileMenuOpen(false)}>
              Sign in to SONIC <ArrowRight size={16} />
            </Link>
          </nav>
        )}
      </header>

      <section className="sonic-hero relative mx-auto grid max-w-[1320px] items-center gap-5 px-5 pb-16 pt-12 sm:px-8 md:min-h-[690px] md:grid-cols-[1.02fr_0.98fr] md:pb-20 md:pt-16">
        <div className="sonic-hero-copy relative z-10">
          <div className="sonic-status-pill">
            <span className={`sonic-status-dot sonic-status-${systemStatus}`} />
            <span>{statusLabel}</span>
            <span className="sonic-status-divider" />
            <span>AI WORKSTATION</span>
          </div>

          <h1 className="sonic-headline mt-8 max-w-[720px] text-[clamp(3.2rem,7.6vw,6.65rem)] font-semibold leading-[0.98] tracking-[-0.075em]">
            Give AI a task.
            <br />
            <span className="sonic-headline-accent">It gets to work.</span>
          </h1>
          <p className="mt-7 max-w-[570px] text-base leading-7 text-[#a7b0bd] sm:text-lg sm:leading-8">
            SONIC is an AI teammate with its own computer. It can see the desktop, use the terminal,
            coordinate specialist agents, and verify what actually happened.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link href="/login" className="sonic-button sonic-button-primary">
              Sign in and get started <ArrowRight size={17} />
            </Link>
            <a href="#how-it-works" className="sonic-button sonic-button-quiet">
              <Play size={15} /> See how it works
            </a>
          </div>

          <div className="mt-9 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] font-medium tracking-wide text-[#77818e]">
            <span className="inline-flex items-center gap-2"><CircleDot size={13} className="text-[#75dfd1]" /> GUI + terminal</span>
            <span className="sonic-proof-separator" />
            <span className="inline-flex items-center gap-2"><Layers3 size={13} className="text-[#ae9cff]" /> Parallel agents</span>
            <span className="sonic-proof-separator" />
            <span className="inline-flex items-center gap-2"><ShieldCheck size={13} className="text-[#ff9d85]" /> Evidence-led</span>
          </div>
        </div>

        <div className="sonic-hero-art relative mx-auto h-[390px] w-full max-w-[620px] sm:h-[490px] md:h-[590px]" role="img" aria-label="Animated visualization of SONIC observing, coordinating agents, and verifying outcomes">
          <div className="sonic-art-grid" aria-hidden="true" />
          <div className="sonic-art-glow" aria-hidden="true" />
          <HeroScene />
          <div className="sonic-orbit-label sonic-orbit-label-top">
            <span className="sonic-orbit-label-icon sonic-mint"><Eye size={14} /></span>
            <span><strong>OBSERVE</strong><small>desktop + context</small></span>
          </div>
          <div className="sonic-orbit-label sonic-orbit-label-right">
            <span className="sonic-orbit-label-icon sonic-violet"><Workflow size={14} /></span>
            <span><strong>COORDINATE</strong><small>focused agents</small></span>
          </div>
          <div className="sonic-orbit-label sonic-orbit-label-bottom">
            <span className="sonic-orbit-label-icon sonic-coral"><FileCheck2 size={14} /></span>
            <span><strong>VERIFY</strong><small>observable results</small></span>
          </div>
          <div className="sonic-core-caption">
            <span className="sonic-caption-pulse" />
            <span>ONE GOAL · MANY CAPABILITIES</span>
          </div>
          <div className="sonic-floating-note">
            <span className="sonic-note-icon"><Command size={14} /></span>
            <span><strong>Computer-native AI</strong><small>Not just chat. Real work.</small></span>
          </div>
        </div>

        <a href="#capabilities" className="sonic-scroll-cue" aria-label="Scroll to capabilities">
          <ArrowDown size={15} /> EXPLORE SONIC
        </a>
      </section>

      <section className="sonic-trust-strip" aria-label="SONIC principles">
        <div className="mx-auto grid max-w-[1320px] gap-5 px-5 py-6 sm:grid-cols-3 sm:px-8 sm:py-7">
          {PRINCIPLES.map(({ icon: Icon, title, text }) => (
            <div className="sonic-trust-item" key={title}>
              <span className="sonic-trust-icon"><Icon size={17} /></span>
              <span><strong>{title}</strong><small>{text}</small></span>
            </div>
          ))}
        </div>
      </section>

      <section id="capabilities" className="sonic-section mx-auto max-w-[1320px] px-5 py-24 sm:px-8 md:py-32">
        <div className="sonic-section-heading">
          <span className="sonic-eyebrow"><Sparkles size={13} /> BEYOND THE CHAT WINDOW</span>
          <h2>AI that can work<br className="sm:hidden" /> <span className="sonic-muted-heading">where the work is.</span></h2>
          <p>SONIC connects perception, action, and verification in one computer-native workspace.</p>
        </div>

        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {CAPABILITIES.map(({ icon: Icon, label, title, description, accent }, index) => (
            <article className={`sonic-capability-card sonic-card-${accent}`} key={title} style={{ animationDelay: `${index * 100}ms` }}>
              <div className="sonic-card-topline"><span>{label}</span><span>0{index + 1}</span></div>
              <div className="sonic-capability-icon"><Icon size={21} strokeWidth={1.8} /></div>
              <h3>{title}</h3>
              <p>{description}</p>
              <span className="sonic-card-arrow" aria-hidden="true"><ArrowUpRight size={17} /></span>
            </article>
          ))}
        </div>
      </section>

      <section id="how-it-works" className="sonic-how-section">
        <div className="sonic-section mx-auto max-w-[1320px] px-5 py-24 sm:px-8 md:py-32">
          <div className="sonic-how-heading">
            <div>
              <span className="sonic-eyebrow"><Orbit size={13} /> A CLOSED-LOOP WORKFLOW</span>
              <h2>From instruction<br />to <span className="sonic-muted-heading">verified outcome.</span></h2>
            </div>
            <p>SONIC observes, acts, checks the result, and adapts when reality differs from the plan.</p>
          </div>
          <div className="sonic-workflow-grid">
            {WORKFLOW.map(({ number, icon: Icon, title, description, detail }, index) => (
              <article className="sonic-workflow-card" key={number}>
                <div className="sonic-workflow-card-head">
                  <span className="sonic-workflow-number">{number}</span>
                  <span className="sonic-workflow-icon"><Icon size={18} /></span>
                  {index < WORKFLOW.length - 1 && <span className="sonic-workflow-connector" aria-hidden="true"><ChevronRight size={17} /></span>}
                </div>
                <h3>{title}</h3>
                <p>{description}</p>
                <div className="sonic-workflow-detail"><span />{detail}</div>
              </article>
            ))}
          </div>

          <div className="sonic-parallel-panel">
            <div className="sonic-parallel-copy">
              <span className="sonic-eyebrow"><Radio size={13} /> BUILT FOR PARALLEL WORK</span>
              <h3>Different agents.<br />One coordinated effort.</h3>
              <p>When a task benefits from parallel work, SONIC can assign distinct roles and show what each worker is doing—on the desktop or in code.</p>
              <Link href="/about" className="sonic-inline-link">Explore the architecture <ArrowRight size={15} /></Link>
            </div>
            <div className="sonic-worker-stack" aria-label="Illustrative parallel worker roles">
              <div className="sonic-worker-card">
                <span className="sonic-worker-avatar sonic-worker-mint"><Monitor size={16} /></span>
                <span className="sonic-worker-meta"><strong>Desktop operator</strong><small>GUI · Browser interaction</small></span>
                <span className="sonic-worker-state"><span /> COMPUTER</span>
              </div>
              <div className="sonic-worker-card sonic-worker-offset">
                <span className="sonic-worker-avatar sonic-worker-violet"><Code2 size={16} /></span>
                <span className="sonic-worker-meta"><strong>Software builder</strong><small>Terminal · Code changes</small></span>
                <span className="sonic-worker-state"><span /> CODE</span>
              </div>
              <div className="sonic-worker-card sonic-worker-offset-more">
                <span className="sonic-worker-avatar sonic-worker-coral"><FileCheck2 size={16} /></span>
                <span className="sonic-worker-meta"><strong>Independent verifier</strong><small>Checks · Evidence</small></span>
                <span className="sonic-worker-state"><span /> VERIFY</span>
              </div>
              <div className="sonic-worker-caption"><Bot size={14} /> EXAMPLE ROLES · ASSIGNED PER TASK</div>
            </div>
          </div>
        </div>
      </section>

      <section className="sonic-cta-section mx-auto max-w-[1320px] px-5 py-24 sm:px-8 md:py-32">
        <div className="sonic-cta-panel">
          <div className="sonic-cta-orb" aria-hidden="true" />
          <span className="sonic-eyebrow"><Sparkles size={13} /> YOUR WORKSPACE IS READY</span>
          <h2>Give the work<br />a place to happen.</h2>
          <p>Sign in to meet SONIC at its workstation.</p>
          <Link href="/login" className="sonic-button sonic-button-primary">
            Continue to sign in <ArrowRight size={17} />
          </Link>
          <div className="sonic-cta-footnote"><LockKeyhole size={12} /> Secure access · Dedicated workstation</div>
        </div>
      </section>

      <footer className="sonic-footer">
        <div className="mx-auto flex max-w-[1320px] flex-col gap-5 px-5 py-7 sm:px-8 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <BrandMark withWordmark href="/" />
            <span className="sonic-footer-divider" />
            <span className="sonic-footer-caption">A computer-native AI teammate.</span>
          </div>
          <nav className="flex flex-wrap items-center gap-x-6 gap-y-2" aria-label="Footer navigation">
            <Link href="/about">About</Link>
            <Link href="/login">Sign in</Link>
            <Link href="/workstation">Workstation</Link>
          </nav>
          <span className="sonic-footer-caption">© {new Date().getFullYear()} SONIC</span>
        </div>
      </footer>
    </main>
  );
}
