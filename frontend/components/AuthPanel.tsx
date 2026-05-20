"use client";

import { useState, type FormEvent } from "react";
import { Eye, EyeOff, Activity, Shield, BarChart3, Zap, ArrowRight } from "lucide-react";
import { loginUser, registerUser } from "../lib/api";
import type { User } from "../lib/types";
import { cn } from "@/lib/utils";

type Props = { onAuthenticated: (token: string, user: User) => void };

const FEATURES = [
  { icon: Activity,  text: "Live face & voice stress analysis" },
  { icon: BarChart3, text: "Trend insights across all sessions" },
  { icon: Shield,    text: "Private by design — data stays yours" },
  { icon: Zap,       text: "Results delivered in under 10 seconds" },
];

export function AuthPanel({ onAuthenticated }: Props) {
  const [mode, setMode]     = useState<"login" | "register">("login");
  const [name, setName]     = useState("");
  const [uname, setUname]   = useState("");
  const [email, setEmail]   = useState("");
  const [pwd, setPwd]       = useState("");
  const [show, setShow]     = useState(false);
  const [err, setErr]       = useState<string | null>(null);
  const [busy, setBusy]     = useState(false);

  function switchMode(m: "login" | "register") {
    setMode(m); setName(""); setUname(""); setEmail(""); setPwd("");
    setShow(false); setErr(null); setBusy(false);
  }

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault(); setBusy(true); setErr(null);
    try {
      const res = mode === "register"
        ? await registerUser({ name, username: uname, email, password: pwd })
        : await loginUser({ email, password: pwd });
      onAuthenticated(res.token, res.user);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Authentication failed.");
    } finally { setBusy(false); }
  }

  return (
    <div className="auth-root">

      {/* ══ LEFT ══ */}
      <aside className="auth-left">
        <div className="auth-left-ring" aria-hidden />

        <div className="auth-brand">
          <div className="auth-logo">M</div>
          <span className="auth-wordmark">MindPulse</span>
        </div>

        <div className="auth-hero">
          <span className="auth-pill">
            <span className="auth-pill-dot" />
            AI-powered wellbeing
          </span>

          <h1 className="auth-h1">
            Know your<br />
            <em>stress level</em><br />
            instantly.
          </h1>

          <p className="auth-lead">
            MindPulse reads facial expressions and voice tone to deliver
            a private, real-time stress reading — no wearable needed.
          </p>

          <ul className="auth-features" role="list">
            {FEATURES.map(({ icon: Icon, text }) => (
              <li key={text} className="auth-feat">
                <span className="auth-feat-icon">
                  <Icon size={16} strokeWidth={2.2} />
                </span>
                {text}
              </li>
            ))}
          </ul>

          <p style={{
            marginTop: 52, display: "flex", alignItems: "center", gap: 10,
            color: "rgba(255,255,255,0.25)", fontSize: "0.79rem", fontWeight: 500,
          }}>
            <ArrowRight size={13} style={{ color: "var(--green-hi)", flexShrink: 0 }} />
            Sign in on the right to get started
          </p>
        </div>

        <p className="auth-copy">© {new Date().getFullYear()} MindPulse. All rights reserved.</p>
      </aside>

      {/* ══ RIGHT ══ */}
      <div className="auth-right">
        <div className="auth-form-wrap">

          {/* mobile brand */}
          <div className="auth-mb-brand">
            <div className="auth-mb-logo">M</div>
            <span className="auth-mb-word">MindPulse</span>
          </div>

          {/* header */}
          <div className="auth-fhead">
            <p className="auth-fsuper">
              {mode === "login" ? "Welcome back" : "Get started free"}
            </p>
            <h2 className="auth-ftitle">
              {mode === "login" ? "Sign in to\nMindPulse" : "Create your\naccount"}
            </h2>
            <p className="auth-fsub">
              {mode === "login"
                ? "Access your dashboard and reading history."
                : "Your private wellbeing workspace awaits."}
            </p>
          </div>

          {/* tabs */}
          <div className="auth-tabs" role="tablist">
            {(["login", "register"] as const).map((m) => (
              <button key={m} type="button" role="tab"
                aria-selected={mode === m}
                onClick={() => switchMode(m)}
                className={cn("auth-tab", mode === m && "on")}>
                {m === "login" ? "Sign in" : "Create account"}
              </button>
            ))}
          </div>

          {/* fields */}
          <form onSubmit={onSubmit} className="auth-fields">
            {mode === "register" && (
              <>
                <div className="auth-field">
                  <label className="auth-flabel" htmlFor="a-name">Full name</label>
                  <input id="a-name" className="auth-input" placeholder="Jane Smith"
                    value={name} onChange={e => setName(e.target.value)}
                    minLength={2} required autoComplete="name" />
                </div>
                <div className="auth-field">
                  <label className="auth-flabel" htmlFor="a-user">Username</label>
                  <input id="a-user" className="auth-input" placeholder="janesmith"
                    value={uname} onChange={e => setUname(e.target.value)}
                    minLength={3} pattern="[a-zA-Z0-9_]+" required autoComplete="username" />
                </div>
              </>
            )}

            <div className="auth-field">
              <label className="auth-flabel" htmlFor="a-email">Email address</label>
              <input id="a-email" type="email" className="auth-input" placeholder="you@example.com"
                value={email} onChange={e => setEmail(e.target.value)}
                required autoComplete="email" />
            </div>

            <div className="auth-field">
              <div className="auth-flabel-row">
                <label className="auth-flabel" htmlFor="a-pwd">Password</label>
                {mode === "login" && (
                  <button type="button" className="auth-forgot">Forgot password?</button>
                )}
              </div>
              <div className="auth-pw-wrap">
                <input id="a-pwd" type={show ? "text" : "password"}
                  className="auth-input auth-input-pw"
                  placeholder={mode === "login" ? "Your password" : "Min. 8 characters"}
                  value={pwd} onChange={e => setPwd(e.target.value)}
                  minLength={8} required
                  autoComplete={mode === "login" ? "current-password" : "new-password"} />
                <button type="button" aria-label={show ? "Hide password" : "Show password"}
                  onClick={() => setShow(v => !v)} className="auth-eye">
                  {show ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {err && (
              <div role="alert" className="a-err">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round"
                  strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 1 }}>
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                {err}
              </div>
            )}

            <button type="submit" disabled={busy} className="auth-btn">
              {busy ? "Please wait…" : mode === "login" ? "Sign in →" : "Create account →"}
            </button>
          </form>

          {/* divider + switch */}
          <div className="auth-divider">
            <div className="auth-divider-line" />
            <span className="auth-divider-text">
              {mode === "login" ? "No account yet?" : "Already signed up?"}
            </span>
            <div className="auth-divider-line" />
          </div>

          <div className="auth-switch">
            <button type="button" onClick={() => switchMode(mode === "login" ? "register" : "login")}
              className="auth-switch-btn">
              {mode === "login" ? "Create a free account" : "Sign in instead"}
              <ArrowRight size={13} />
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}
