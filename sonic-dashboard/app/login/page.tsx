"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ShieldCheck,
  Sparkles,
  ArrowRight,
  AlertCircle,
  Terminal,
  Cpu,
  Lock,
  ExternalLink,
  Info,
} from "lucide-react";
import Link from "next/link";
import { setAuthToken } from "../../lib/auth";
import { API_BASE } from "../../lib/api";
import { BrandMark } from "../../components/common/BrandMark";

function LoginContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [googleConfig, setGoogleConfig] = useState<{ enabled: boolean; clientId: string | null }>({
    enabled: false,
    clientId: null,
  });
  const [configChecked, setConfigChecked] = useState(false);
  const [localEmail, setLocalEmail] = useState("");

  // Check if returning from Google OAuth callback with a token in query params
  useEffect(() => {
    const token = searchParams.get("token");
    if (token) {
      const email = searchParams.get("email") || "operator@local";
      const name = searchParams.get("name") || email.split("@")[0];
      const role = searchParams.get("role") || "operator";
      const tenantId = searchParams.get("tenant_id") || "default";

      setAuthToken(token, {
        email,
        name,
        role,
        tenant_id: tenantId,
      });
      router.push("/");
      return;
    }

    // Probe server Google OAuth configuration status
    const probeGoogleConfig = async () => {
      try {
        const res = await fetch(`${API_BASE}/auth/google/config`);
        if (res.ok) {
          const data = await res.json();
          setGoogleConfig({
            enabled: Boolean(data?.enabled),
            clientId: data?.client_id || null,
          });
        }
      } catch {
        setGoogleConfig({ enabled: false, clientId: null });
      } finally {
        setConfigChecked(true);
      }
    };

    probeGoogleConfig();
  }, [searchParams, router]);

  // Real Google OAuth Flow — redirects browser directly to Google consent screen
  const handleRealGoogleSignIn = () => {
    setLoading(true);
    setError(null);

    if (!googleConfig.enabled) {
      setError("Google OAuth is not configured on the backend. Please configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in the environment.");
      setLoading(false);
      return;
    }

    const returnUrl = encodeURIComponent(window.location.origin + "/login");
    window.location.href = `${API_BASE}/auth/google/login?redirect_url=${returnUrl}`;
  };

  // Local Workstation Operator Access (Development / Sandbox mode)
  const handleLocalSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const emailToUse = localEmail.trim() || "operator@local";
    const nameToUse = emailToUse.split("@")[0].replace(/[._-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: emailToUse,
          name: nameToUse,
          role: "operator",
          tenant_id: "default",
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData?.detail || `Local authentication failed with status ${res.status}`);
      }

      const data = await res.json();
      if (data?.access_token) {
        setAuthToken(data.access_token, {
          email: data.user?.email || emailToUse,
          name: data.user?.name || nameToUse,
          role: data.user?.role || "operator",
          tenant_id: data.user?.tenant_id || "default",
        });
        router.push("/");
      } else {
        throw new Error("No authorization token was issued by the workstation.");
      }
    } catch (err: any) {
      console.error("Local sign-in error:", err);
      setError(err.message || "Failed to complete local workstation authentication.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-ink-950 text-slate-200 flex flex-col justify-between font-sans relative overflow-hidden select-none">
      {/* Dynamic ambient cyber glow background */}
      <div className="fixed inset-0 bg-grid-glow pointer-events-none opacity-80" />
      <div className="absolute top-[-15%] left-[20%] w-[650px] h-[650px] bg-primary-600/10 rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute bottom-[-15%] right-[15%] w-[650px] h-[650px] bg-secondary-600/15 rounded-full blur-[160px] pointer-events-none" />

      {/* Top minimalistic header */}
      <header className="h-16 border-b border-ink-800/60 backdrop-blur-xl bg-ink-950/70 px-8 flex items-center justify-between z-10">
        <BrandMark withWordmark href="/landing" />
        <div className="flex items-center gap-4 text-xs font-mono">
          <Link href="/landing" className="text-muted hover:text-white transition flex items-center gap-1.5">
            <span>Architecture Overview</span>
            <ExternalLink className="w-3 h-3 opacity-60" />
          </Link>
          <div className="h-4 w-[1px] bg-ink-750" />
          <div className="inline-flex items-center gap-1.5 text-[11px] text-secondary-400 bg-secondary-500/10 px-2.5 py-1 rounded-full border border-secondary-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-secondary-400 animate-pulse" />
            <span>ASI System Online</span>
          </div>
        </div>
      </header>

      {/* Centerpiece Sign-In Card */}
      <main className="flex-1 flex items-center justify-center p-6 z-10">
        <div className="w-full max-w-[440px] glass-card rounded-2xl p-8 sm:p-9 shadow-2xl border border-ink-700/60 bg-ink-900/80 backdrop-blur-2xl space-y-6 relative animate-fade-in-up">
          
          {/* Header */}
          <div className="space-y-3 text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ink-800/90 border border-ink-700 text-slate-300 text-[11px] font-mono shadow-sm">
              <Sparkles className="w-3.5 h-3.5 text-primary-400" />
              <span>SONIC AUTONOMOUS WORKSTATION</span>
            </div>

            <h1 className="text-2xl font-extrabold tracking-tight text-white">
              Sign in to SONIC
            </h1>
            <p className="text-xs text-muted leading-relaxed font-mono">
              Operate alongside your autonomous AI-Human digital colleague with full dual-plane computer workstation agency.
            </p>
          </div>

          {error && (
            <div className="p-3.5 rounded-xl bg-danger/10 border border-danger/30 text-danger text-xs font-mono flex items-center gap-2.5 animate-fade-in-up">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span className="leading-snug">{error}</span>
            </div>
          )}

          {/* Authentication Actions */}
          <div className="space-y-4">
            
            {/* Primary Google OAuth Action */}
            <button
              onClick={handleRealGoogleSignIn}
              disabled={loading || (!googleConfig.enabled && configChecked)}
              className={`group relative w-full flex items-center justify-center gap-3.5 px-6 py-3.5 rounded-xl font-semibold text-sm shadow-lg transition-all duration-200 border cursor-pointer ${
                googleConfig.enabled
                  ? "bg-white hover:bg-slate-50 text-slate-900 border-slate-200 active:scale-[0.99]"
                  : "bg-ink-800/60 text-slate-400 border-ink-700/50 cursor-not-allowed opacity-60"
              }`}
            >
              {loading ? (
                <div className="flex items-center gap-2.5 text-slate-700 font-medium">
                  <span className="w-4 h-4 border-2 border-slate-400 border-t-slate-800 rounded-full animate-spin" />
                  <span>Connecting to Google OAuth...</span>
                </div>
              ) : (
                <>
                  {/* Official Google 4-Color 'G' SVG */}
                  <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24">
                    <path
                      fill="#4285F4"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="#34A853"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="#FBBC05"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                    />
                    <path
                      fill="#EA4335"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                    />
                  </svg>
                  <span className={`tracking-tight font-bold ${googleConfig.enabled ? "text-slate-800 group-hover:text-black" : "text-slate-400"}`}>
                    {googleConfig.enabled ? "Continue with Google" : "Google OAuth (Not Configured)"}
                  </span>
                  {googleConfig.enabled && (
                    <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-slate-700 group-hover:translate-x-0.5 transition-transform ml-auto" />
                  )}
                </>
              )}
            </button>

            {/* Google OAuth Status telemetry */}
            {configChecked && !googleConfig.enabled && (
              <div className="p-3 rounded-xl bg-ink-950/90 border border-ink-800 text-[11px] font-mono text-muted space-y-1">
                <div className="flex items-center gap-1.5 text-amber-400/90 font-medium">
                  <Info className="w-3.5 h-3.5 flex-shrink-0" />
                  <span>Google OAuth Unconfigured</span>
                </div>
                <p className="text-[10px] text-muted-dim leading-relaxed">
                  Set <code className="text-slate-300">GOOGLE_CLIENT_ID</code> and <code className="text-slate-300">GOOGLE_CLIENT_SECRET</code> in the backend environment to enable live Google Sign-In.
                </p>
              </div>
            )}

            {/* Local Workstation Operator Section */}
            <div className="pt-2">
              <div className="relative flex items-center justify-center my-3">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-ink-800" />
                </div>
                <span className="relative px-3 bg-ink-900 text-[10px] uppercase tracking-wider font-mono text-muted-dim">
                  Operator Direct Access
                </span>
              </div>

              <form onSubmit={handleLocalSignIn} className="space-y-2.5">
                <div>
                  <label className="block text-[11px] font-mono text-muted mb-1">
                    Operator Email
                  </label>
                  <input
                    type="email"
                    value={localEmail}
                    onChange={(e) => setLocalEmail(e.target.value)}
                    placeholder="operator@local"
                    className="input-field !text-xs !py-2.5 font-mono w-full"
                    disabled={loading}
                  />
                </div>
                <button
                  type="submit"
                  disabled={loading}
                  className="btn-primary w-full !py-2.5 text-xs font-mono font-medium flex items-center justify-center gap-2 cursor-pointer"
                >
                  <Terminal className="w-3.5 h-3.5" />
                  <span>Launch Workstation Session</span>
                </button>
              </form>
            </div>

          </div>

          {/* Separator / Divider */}
          <div className="relative flex items-center justify-center pt-1">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-ink-800" />
            </div>
            <span className="relative px-3 bg-ink-900 text-[10px] uppercase tracking-widest font-mono text-muted-dim">
              Architect Guarantees
            </span>
          </div>

          {/* Key Value & Security Guarantees */}
          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
            <div className="p-2.5 rounded-lg bg-ink-950/60 border border-ink-800/80 flex items-center gap-2">
              <Lock className="w-3.5 h-3.5 text-primary-400 flex-shrink-0" />
              <span className="text-muted-bright text-[10px] truncate">Zero Password Risk</span>
            </div>
            <div className="p-2.5 rounded-lg bg-ink-950/60 border border-ink-800/80 flex items-center gap-2">
              <Cpu className="w-3.5 h-3.5 text-secondary-400 flex-shrink-0" />
              <span className="text-muted-bright text-[10px] truncate">Sealed Sandbox</span>
            </div>
          </div>

          {/* Security Badge */}
          <div className="pt-1 flex items-center justify-center gap-2 text-[11px] font-mono text-muted-dim">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Protected by Hardware-Speed Safety Kernel</span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="h-14 border-t border-ink-800/60 px-8 flex items-center justify-between text-[11px] font-mono text-muted-dim z-10 backdrop-blur-md bg-ink-950/40">
        <div className="flex items-center gap-2">
          <span>SONIC-REDA Core v0.2.0</span>
          <span>•</span>
          <span className="text-slate-400">Autonomous Penetration Architect</span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/landing" className="hover:text-slate-300 transition">
            Documentation
          </Link>
          <Link href="/" className="text-primary-400 hover:text-primary-300 transition flex items-center gap-1">
            <span>Direct Access</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </footer>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-ink-950 flex items-center justify-center">
          <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
        </div>
      }
    >
      <LoginContent />
    </Suspense>
  );
}
