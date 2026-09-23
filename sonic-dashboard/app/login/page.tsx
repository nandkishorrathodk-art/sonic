"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useRouter } from "next/navigation";
import {
  ShieldCheck,
  Sparkles,
  ArrowRight,
  AlertCircle,
  Terminal,
  Cpu,
  Lock,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import Link from "next/link";
import { setAuthToken } from "../../lib/auth";
import { API_BASE } from "../../lib/api";
import { BrandMark } from "../../components/common/BrandMark";

function LoginContent() {
  const router = useRouter();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [customGoogleEmail, setCustomGoogleEmail] = useState("");
  const [showCustomInput, setShowCustomInput] = useState(false);

  // Default recognized Google identity for instant 1-click dev authentication
  const defaultGoogleAccount = {
    email: "nandkishorrathodk@gmail.com",
    name: "Nandkishor Rathod",
    avatarInitial: "N",
  };

  // Check if returning from Google OAuth with a token in the URL fragment.
  useEffect(() => {
    const fragment = typeof window !== "undefined" ? window.location.hash.slice(1) : "";
    const fragmentParams = new URLSearchParams(fragment);
    const token = fragmentParams.get("token");
    if (token) {
      const email = fragmentParams.get("email") || defaultGoogleAccount.email;
      const name = fragmentParams.get("name") || defaultGoogleAccount.name;
      const role = fragmentParams.get("role") || "operator";
      const tenantId = fragmentParams.get("tenant_id") || "default";

      setAuthToken(token, {
        email,
        name,
        role,
        tenant_id: tenantId,
      });
      router.push("/");
    }
  }, [router]);

  const handleGoogleSignIn = async (targetEmail?: string) => {
    setLoading(true);
    setError(null);

    const emailToUse = targetEmail || customGoogleEmail || defaultGoogleAccount.email;
    const nameToUse = targetEmail === defaultGoogleAccount.email ? defaultGoogleAccount.name : emailToUse.split("@")[0].replace(/[._-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    try {
      // 1. Check if backend has live Google OAuth configured
      let oauthConfig = { enabled: false };
      try {
        const configRes = await fetch(`${API_BASE}/auth/google/config`);
        if (configRes.ok) {
          oauthConfig = await configRes.json();
        }
      } catch {
        // backend offline or endpoint pending, fall back to direct dev auth
      }

      // 2. If live Google OAuth is enabled in production, redirect to Google consent screen
      if (oauthConfig.enabled) {
        const returnUrl = encodeURIComponent("/login");
        window.location.href = `${API_BASE}/auth/google/login?redirect_url=${returnUrl}`;
        return;
      }

      // 3. Otherwise (Development / Local mode): Direct instant Google authentication
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
        throw new Error(`Authentication server returned status ${res.status}`);
      }

      const data = await res.json();
      if (data?.access_token) {
        setAuthToken(data.access_token, {
          email: data.user?.email || emailToUse,
          name: data.user?.name || nameToUse,
          role: data.user?.role || "operator",
          tenant_id: data.user?.tenant_id || "default",
        });

        // Instant smooth transition into the workstation
        router.push("/");
      } else {
        throw new Error("No authorization token was issued by the server.");
      }
    } catch (err: any) {
      console.error("Google sign-in error:", err);
      setError(err.message || "Failed to complete Google authentication.");
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

      {/* Modern Centerpiece Sign-In Card */}
      <main className="flex-1 flex items-center justify-center p-6 z-10">
        <div className="w-full max-w-[440px] glass-card rounded-2xl p-8 sm:p-9 shadow-2xl border border-ink-700/60 bg-ink-900/80 backdrop-blur-2xl space-y-7 relative animate-fade-in-up">
          
          {/* Header & Logo */}
          <div className="space-y-3 text-center">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ink-800/90 border border-ink-700 text-slate-300 text-[11px] font-mono shadow-sm">
              <Sparkles className="w-3.5 h-3.5 text-primary-400" />
              <span>SONIC AUTONOMOUS WORKSTATION</span>
            </div>

            <h1 className="text-2xl font-extrabold tracking-tight text-white">
              Sign in to SONIC
            </h1>
            <p className="text-xs text-muted leading-relaxed font-mono">
              Empower your autonomous AI penetration architect with dedicated computer agency.
            </p>
          </div>

          {error && (
            <div className="p-3.5 rounded-xl bg-danger/10 border border-danger/30 text-danger text-xs font-mono flex items-center gap-2.5 animate-fade-in-up">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span className="leading-snug">{error}</span>
            </div>
          )}

          {/* Core Action: Modern Google Sign-In */}
          <div className="space-y-4">
            
            {/* Primary Google Sign-In Button */}
            <button
              onClick={() => handleGoogleSignIn()}
              disabled={loading}
              className="group relative w-full flex items-center justify-center gap-3.5 px-6 py-3.5 rounded-xl bg-white hover:bg-slate-50 text-slate-900 font-semibold text-sm shadow-lg hover:shadow-xl transition-all duration-200 border border-slate-200 active:scale-[0.99] disabled:opacity-70 disabled:cursor-not-allowed cursor-pointer"
            >
              {loading ? (
                <div className="flex items-center gap-2.5 text-slate-700 font-medium">
                  <span className="w-4 h-4 border-2 border-slate-400 border-t-slate-800 rounded-full animate-spin" />
                  <span>Connecting with Google...</span>
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
                  <span className="tracking-tight text-slate-800 font-bold group-hover:text-black">
                    Continue with Google
                  </span>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-slate-700 group-hover:translate-x-0.5 transition-transform ml-auto" />
                </>
              )}
            </button>

            {/* Google Profile Card Preview */}
            <div className="p-3 rounded-xl bg-ink-950/80 border border-ink-800/90 flex items-center justify-between text-xs">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-primary-500 to-secondary-500 p-[1.5px] flex-shrink-0">
                  <div className="w-full h-full rounded-full bg-ink-950 flex items-center justify-center font-bold text-white text-xs">
                    {defaultGoogleAccount.avatarInitial}
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="font-semibold text-white truncate text-[12px] flex items-center gap-1.5">
                    <span>{defaultGoogleAccount.name}</span>
                    <CheckCircle2 className="w-3 h-3 text-secondary-400 flex-shrink-0" />
                  </div>
                  <div className="text-[11px] text-muted-dim font-mono truncate">
                    {defaultGoogleAccount.email}
                  </div>
                </div>
              </div>

              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Verified
              </span>
            </div>

            {/* Custom Google Account Switcher (Optional) */}
            <div className="pt-1">
              {!showCustomInput ? (
                <button
                  type="button"
                  onClick={() => setShowCustomInput(true)}
                  className="w-full text-center text-[11px] text-muted hover:text-slate-300 font-mono transition py-1 cursor-pointer"
                >
                  Use a different Google / Workspace email?
                </button>
              ) : (
                <div className="space-y-2 pt-1 animate-fade-in-up">
                  <div className="flex items-center gap-2">
                    <input
                      type="email"
                      value={customGoogleEmail}
                      onChange={(e) => setCustomGoogleEmail(e.target.value)}
                      placeholder="your.google.account@gmail.com"
                      className="input-field !text-xs !py-2 font-mono"
                    />
                    <button
                      type="button"
                      disabled={loading || !customGoogleEmail}
                      onClick={() => handleGoogleSignIn(customGoogleEmail)}
                      className="btn-secondary !px-3 !py-2 text-xs flex-shrink-0"
                    >
                      Sign In
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowCustomInput(false)}
                    className="text-[10px] text-muted-dim hover:text-muted font-mono"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>

          </div>

          {/* Separator / Divider */}
          <div className="relative flex items-center justify-center">
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
