"use client";

import { useMemo, useRef, useState, useEffect, type ReactNode } from "react";
import {
  LayoutDashboard, Activity, Clock, User2, LogOut,
  Camera, StopCircle, ScanLine, TrendingUp, AlertTriangle, CheckCircle,
  Search, Filter, Eye, EyeOff, Save, KeyRound, Brain, ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { analyzeSample, changePassword, updateProfile } from "@/lib/api";
import { captureVideoFrames, delay, recordAudioSample, waitForVideoReady } from "@/lib/capture";
import type { DashboardSummary, InferenceResult, StressLevel, User } from "@/lib/types";
import { cn } from "@/lib/utils";

type Props = {
  user: User; token: string; summary: DashboardSummary | null;
  onRefresh: () => Promise<void>; refreshing: boolean;
  onUserUpdated: (u: User) => void; onLogout: () => void;
};
type View = "overview" | "session" | "history" | "profile";

const SCAN_READY_MS  = 900;
const SCAN_REC_MS    = 5500;
const SCAN_FRAMES    = 5;
const SCAN_FRAME_INT = 850;
const SCAN_TOTAL_MS  = SCAN_READY_MS + SCAN_REC_MS;

const pct = (v: number) => `${Math.round(v * 100)}%`;
const cap = (s: string)  => s.charAt(0).toUpperCase() + s.slice(1);

function stressMeta(lvl: StressLevel) {
  if (lvl === "high")   return { title: "Needs a pause",  badge: "High",   summary: "This check-in suggests you may be feeling quite strained right now.", guidance: "Take a short break, slow your breathing, and check in again after a moment.", feeling: "You may be under quite a bit of pressure right now." };
  if (lvl === "medium") return { title: "A bit tense",    badge: "Medium", summary: "This check-in shows some tension or pressure at the moment.",           guidance: "A short break or a calmer environment may help before the next check-in.", feeling: "There are some signs of tension, but nothing extreme." };
  return                       { title: "Steady & calm",  badge: "Low",    summary: "This check-in suggests a calmer, more stable state.",                    guidance: "Keep going and scan again whenever you want an update.",                   feeling: "You seem fairly steady at the moment." };
}

function heroHeadline(r: InferenceResult | null) {
  if (!r)                     return "Ready for your first check-in";
  if (r.stress_level === "high")   return "You may need a short pause";
  if (r.stress_level === "medium") return "You seem a bit tense right now";
  return "You seem fairly steady right now";
}

const faceObs: Record<string, string>  = { calm: "Your face looked calm and settled.", neutral: "Your face looked fairly steady.", happy: "Your face showed positive energy.", sad: "Your face showed signs of low mood.", angry: "Your face showed visible tension.", stressed: "Your face showed visible stress." };
const voiceObs: Record<string, string> = { calm: "Your voice sounded calm.", neutral: "Your voice sounded steady.", happy: "Your voice sounded upbeat.", sad: "Your voice sounded low or subdued.", angry: "Your voice sounded tense.", stressed: "Your voice sounded pressured." };

const orbCls:  Record<StressLevel, string> = { low: "orb-low", medium: "orb-mid", high: "orb-hi" };
const pillCls: Record<StressLevel, string> = { low: "pill-low", medium: "pill-mid", high: "pill-hi" };
const barCol:  Record<StressLevel, string> = { low: "var(--green)", medium: "var(--amber)", high: "var(--red)" };

function empty(): DashboardSummary {
  return { total_results: 0, latest_result: null, stress_distribution: { low: 0, medium: 0, high: 0 }, average_face_confidence: 0, average_voice_confidence: 0, high_stress_rate: 0, recent_results: [] };
}

function visible(s: DashboardSummary, uid: number): DashboardSummary {
  const rs   = s.recent_results.filter(r => r.source !== "demo_session" && (r.user_id === uid || r.user_id == null));
  const dist = rs.reduce((d, r) => { d[r.stress_level]++; return d; }, { low: 0, medium: 0, high: 0 } as Record<StressLevel, number>);
  const n    = rs.length;
  return { ...s, total_results: n, latest_result: rs[0] ?? null, stress_distribution: dist, average_face_confidence: n ? rs.reduce((t, r) => t + r.face_confidence, 0) / n : 0, average_voice_confidence: n ? rs.reduce((t, r) => t + r.voice_confidence, 0) / n : 0, high_stress_rate: n ? dist.high / n : 0, recent_results: rs };
}

function Pane({ children }: { children: ReactNode }) {
  const [k, setK] = useState(0);
  useEffect(() => { setK(x => x + 1); }, []);
  return <div key={k} className="view-pane">{children}</div>;
}

/* ════════════════════════════════════════════════════════════════ */
export function Dashboard({ user, token, summary, onRefresh, refreshing, onUserUpdated, onLogout }: Props) {
  const raw   = summary ?? empty();
  const data  = visible(raw, user.id);
  const all   = raw.recent_results;
  const dTot  = Math.max(data.total_results, 1);

  const [view,   setView]   = useState<View>("overview");
  const [riskF,  setRiskF]  = useState<"all" | StressLevel>("all");
  const [query,  setQuery]  = useState("");

  const [pName,  setPName]  = useState(user.name);
  const [pUser,  setPUser]  = useState(user.username);
  const [pEmail, setPEmail] = useState(user.email);
  const [curPwd, setCurPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [showC,  setShowC]  = useState(false);
  const [showN,  setShowN]  = useState(false);
  const [pMsg,   setPMsg]   = useState<string | null>(null);
  const [pErr,   setPErr]   = useState<string | null>(null);
  const [pwMsg,  setPwMsg]  = useState<string | null>(null);
  const [pwErr,  setPwErr]  = useState<string | null>(null);

  const [sesOn,  setSesOn]  = useState(false);
  const [sesBusy,setSesBusy]= useState(false);
  const [sesTxt, setSesTxt] = useState("Camera and microphone are ready when you start a session.");
  const [sesErr, setSesErr] = useState<string | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [scanAt, setScanAt] = useState<string | null>(null);

  const vidRef   = useRef<HTMLVideoElement | null>(null);
  const streamRef= useRef<MediaStream | null>(null);
  const runRef   = useRef(0);

  const filtered = useMemo(() => all.filter(r => {
    const ok = riskF === "all" || r.stress_level === riskF;
    return ok && `${r.face_emotion} ${r.voice_emotion} ${r.source}`.toLowerCase().includes(query.trim().toLowerCase());
  }), [all, query, riskF]);

  const latest  = result ?? data.latest_result;
  const lvl     = latest?.stress_level ?? "low";
  const meta    = stressMeta(lvl);

  async function doScan(stream: MediaStream) {
    if (!vidRef.current) throw new Error("Camera not ready.");
    const id = ++runRef.current;
    setResult(null); setScanAt(new Date().toISOString()); setSesBusy(true);
    setSesTxt("Get ready — keep your face visible and speak naturally.");
    try {
      await waitForVideoReady(vidRef.current); if (runRef.current !== id) return;
      await delay(SCAN_READY_MS);              if (runRef.current !== id) return;
      setSesTxt("Recording. Keep speaking naturally until it finishes.");
      const audioP = recordAudioSample(stream, SCAN_REC_MS);
      const frames = await captureVideoFrames(vidRef.current, SCAN_FRAMES, SCAN_FRAME_INT);
      await delay(Math.max(SCAN_REC_MS - SCAN_FRAMES * SCAN_FRAME_INT, 0)); if (runRef.current !== id) return;
      const audio  = await audioP; if (runRef.current !== id) return;
      setSesTxt("Analysing…");
      const res = await analyzeSample(token, { faceImages: frames, audioFile: audio }); if (runRef.current !== id) return;
      setResult(res); await onRefresh(); setSesTxt("Reading saved. Scan again or stop the session.");
    } finally { if (runRef.current === id) setSesBusy(false); }
  }

  async function startSession() {
    setSesErr(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
      streamRef.current = s;
      if (vidRef.current) { vidRef.current.srcObject = s; await vidRef.current.play(); await waitForVideoReady(vidRef.current); }
      setSesOn(true); setSesTxt("Session started. Keep your face visible and speak naturally.");
      await doScan(s);
    } catch (e) { setSesErr(e instanceof Error ? e.message : "Could not start session."); stopSession(); }
  }

  function stopSession() {
    runRef.current++;
    streamRef.current?.getTracks().forEach(t => t.stop()); streamRef.current = null;
    if (vidRef.current) vidRef.current.srcObject = null;
    setSesOn(false); setSesBusy(false); setSesTxt("Session stopped.");
  }

  async function scanAgain() {
    if (!streamRef.current || sesBusy) return; setSesErr(null);
    try { await doScan(streamRef.current); }
    catch (e) { setSesErr(e instanceof Error ? e.message : "Could not analyse sample."); }
  }

  const navItems: { key: View; label: string; icon: ReactNode }[] = [
    { key: "overview", label: "Overview", icon: <LayoutDashboard size={18} /> },
    { key: "session",  label: "Session",  icon: <Camera size={18} /> },
    { key: "history",  label: "History",  icon: <Clock size={18} /> },
    { key: "profile",  label: "Profile",  icon: <User2 size={18} /> },
  ];

  /* ── helpers ── */
  const HeroBanner = ({ eyebrow, title, sub, right }: { eyebrow: string; title: string; sub: string; right?: ReactNode }) => (
    <div className="hero-banner">
      <div className="hero-inner">
        <p className="hero-eyebrow">{eyebrow}</p>
        <h1 className="hero-title">{title}</h1>
        <p className="hero-sub">{sub}</p>
      </div>
      {right}
    </div>
  );

  /* ════════ render ════════ */
  return (
    <div className="db-root">

      {/* ─── TOPBAR ─── */}
      <header className="db-bar">
        <div className="db-brand">
          <div className="db-logo">M</div>
          <span className="db-wordmark">MindPulse</span>
        </div>

        <nav className="db-nav" aria-label="Main navigation">
          {navItems.map(tab => (
            <button key={tab.key} type="button"
              onClick={() => setView(tab.key)}
              className={cn("db-nav-btn", view === tab.key && "on")}>
              {tab.icon}<span>{tab.label}</span>
            </button>
          ))}
        </nav>

      </header>

      {/* ─── CONTENT ─── */}
      <main className="db-content">

        {/* ══════════════ OVERVIEW ══════════════ */}
        {view === "overview" && (
          <Pane>
            {/* Hero */}
            <div className="hero-banner">
              <div className="hero-inner">
                <p className="hero-eyebrow">{data.total_results > 0 ? "Latest reading" : "Welcome to MindPulse"}</p>
                <h1 className="hero-title">{heroHeadline(latest)}</h1>
                <p className="hero-sub">{latest ? meta.feeling : "MindPulse analyses face and voice to give you a real-time stress check-in. Start whenever you're ready."}</p>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 22, flexWrap: "wrap" }}>
                  {latest && (
                    <span className={cn("pill", pillCls[lvl])}>
                      <span className="pill-dot" />{meta.badge} stress
                    </span>
                  )}
                  {latest && (
                    <span style={{ fontSize: "0.79rem", color: "rgba(255,255,255,0.35)", fontWeight: 500 }}>
                      {new Date(latest.timestamp).toLocaleString()}
                    </span>
                  )}
                  <button type="button" onClick={() => setView("session")} style={{
                    display: "flex", alignItems: "center", gap: 8,
                    padding: "9px 20px", borderRadius: 12,
                    border: "1px solid rgba(48,196,154,0.32)",
                    background: "rgba(48,196,154,0.10)",
                    color: "var(--green-hi)", fontSize: "0.85rem", fontWeight: 600,
                    cursor: "pointer", transition: "background .2s",
                    fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif",
                    letterSpacing: "-0.02em",
                  }}>
                    <Camera size={14} />
                    {data.total_results > 0 ? "Scan again" : "Start first scan"}
                    <ChevronRight size={13} />
                  </button>
                </div>
              </div>
              <div className="orb-wrap">
                <div className={cn("orb", latest ? orbCls[lvl] : "orb-none")}>
                  <div className="orb-inner">
                    <span className="orb-val">{latest ? meta.badge : "–"}</span>
                    <span className="orb-lbl">{latest ? "stress level" : "no data yet"}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Metric strip */}
            <div className="metrics">
              {[
                { label: "Total scans",      value: String(data.total_results),        cap: data.total_results === 1 ? "check-in saved" : "check-ins saved", icon: <Activity size={20} />,     cls: "met-g", delay: "0.06s" },
                { label: "High stress rate", value: pct(data.high_stress_rate),         cap: "of your sessions",   icon: <AlertTriangle size={20} />, cls: "met-r", delay: "0.11s" },
                { label: "Face confidence",  value: pct(data.average_face_confidence),  cap: "average accuracy",   icon: <CheckCircle size={20} />,   cls: "met-g", delay: "0.16s" },
                { label: "Voice confidence", value: pct(data.average_voice_confidence), cap: "average accuracy",   icon: <TrendingUp size={20} />,    cls: "met-a", delay: "0.21s" },
              ].map(m => (
                <div key={m.label} className={cn("met anim-fade-up", m.cls)} style={{ animationDelay: m.delay }}>
                  <div className="met-icon">{m.icon}</div>
                  <p className="met-label">{m.label}</p>
                  <p className="met-val">{m.value}</p>
                  <p className="met-cap">{m.cap}</p>
                </div>
              ))}
            </div>

            {/* Two-col: Interpretation + Distribution */}
            <div className="two-col mb-4">
              <div className="pcard anim-fade-up" style={{ animationDelay: "0.12s" }}>
                <p className="p-eye">Interpretation</p>
                {latest ? (
                  <>
                    <h2 className="p-title">{meta.title}</h2>
                    <p className="p-sub" style={{ marginBottom: 20 }}>{meta.summary}</p>
                    <div className="guide">
                      <p className="guide-lbl">Recommended action</p>
                      <p className="guide-txt">{meta.guidance}</p>
                    </div>
                  </>
                ) : (
                  <>
                    <h2 className="p-title">Your first reading starts here</h2>
                    <p className="p-sub">Once you complete a scan, this section will explain your state in plain language and suggest what to do next.</p>
                  </>
                )}
              </div>

              <div className="pcard anim-fade-up" style={{ animationDelay: "0.18s" }}>
                <p className="p-eye">Distribution</p>
                <h2 className="p-title">Stress breakdown</h2>
                <p className="p-sub" style={{ marginBottom: 24 }}>Across {data.total_results} check-in{data.total_results !== 1 ? "s" : ""}.</p>
                {(["low", "medium", "high"] as StressLevel[]).map(l => {
                  const count = data.stress_distribution[l];
                  return (
                    <div key={l} className="dist-row">
                      <div className="dist-hd">
                        <span className={cn("pill", pillCls[l])}><span className="pill-dot" />{cap(l)}</span>
                        <span className="dist-cnt">{count} session{count !== 1 ? "s" : ""}</span>
                      </div>
                      <div className="bar-track">
                        <span className="bar-fill" style={{ width: `${(count / dTot) * 100}%`, background: barCol[l] }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Signal cards */}
            {latest && (
              <div className="two-col mb-4">
                <div className="sig anim-fade-up" style={{ animationDelay: "0.24s" }}>
                  <p className="sig-lbl">Face signal</p>
                  <p className="sig-val">{cap(latest.face_emotion)}</p>
                  <p className="sig-meta">{faceObs[latest.face_emotion.toLowerCase()] ?? latest.face_emotion}<br /><strong style={{ color: "var(--ink-2)", fontWeight: 700 }}>{pct(latest.face_confidence)} confidence</strong></p>
                </div>
                <div className="sig anim-fade-up" style={{ animationDelay: "0.30s" }}>
                  <p className="sig-lbl">Voice signal</p>
                  <p className="sig-val">{cap(latest.voice_emotion)}</p>
                  <p className="sig-meta">{voiceObs[latest.voice_emotion.toLowerCase()] ?? latest.voice_emotion}<br /><strong style={{ color: "var(--ink-2)", fontWeight: 700 }}>{pct(latest.voice_confidence)} confidence</strong></p>
                </div>
              </div>
            )}

            {/* Recent records */}
            <div className="pcard anim-fade-up" style={{ animationDelay: "0.32s" }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22, flexWrap: "wrap", gap: 10 }}>
                <div><p className="p-eye">Activity</p><h2 className="p-title" style={{ marginBottom: 0 }}>Recent records</h2></div>
                <Badge variant="secondary" style={{ fontSize: "0.71rem", fontWeight: 700 }}>{all.length} total</Badge>
              </div>
              {all.length === 0 ? (
                <div className="empty">
                  <Brain size={36} style={{ color: "var(--ink-4)", opacity: .4 }} />
                  <p>Your recent readings will appear here after your first scan.</p>
                </div>
              ) : (
                <div style={{ display: "grid", gap: 8 }}>
                  {all.slice(0, 8).map((r, i) => (
                    <div key={r.id} className="rec-row anim-fade-up" style={{ animationDelay: `${i * 0.04}s` }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
                        <span className={cn("pill", pillCls[r.stress_level])}><span className="pill-dot" />{r.stress_level}</span>
                        <span className="rec-ts hidden sm:block">{new Date(r.timestamp).toLocaleString()}</span>
                        <span className="rec-ts sm:hidden">{new Date(r.timestamp).toLocaleDateString()}</span>
                      </div>
                      <div style={{ display: "flex", gap: 7, flexShrink: 0 }}>
                        <span className="chip hidden sm:inline">Face {pct(r.face_confidence)}</span>
                        <span className="chip">Voice {pct(r.voice_confidence)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </Pane>
        )}

        {/* ══════════════ SESSION ══════════════ */}
        {view === "session" && (
          <Pane>
            <div className="hero-banner" style={{ marginBottom: 20, alignItems: "center" }}>
              <div className="hero-inner">
                <p className="hero-eyebrow">Live capture</p>
                <h1 className="hero-title">{sesOn ? "Session running" : "Start a scan"}</h1>
                <p className="hero-sub">{sesTxt}</p>
              </div>
              {sesOn && (
                <div style={{
                  display: "flex", alignItems: "center", gap: 9, flexShrink: 0,
                  padding: "10px 18px", borderRadius: 999,
                  background: "rgba(48,196,154,0.10)", border: "1px solid rgba(48,196,154,0.28)",
                  color: "var(--green-hi)", fontSize: "0.76rem", fontWeight: 700,
                  letterSpacing: "0.07em",
                  fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif",
                }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--green-hi)", boxShadow: "0 0 10px var(--green-hi)", animation: "dot-breathe 2.2s ease-in-out infinite", display: "inline-block" }} />
                  LIVE
                </div>
              )}
            </div>

            <div className="ses-grid">
              {/* Controls */}
              <div className="pcard" style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                <p style={{ fontSize: "0.82rem", color: "var(--ink-4)", lineHeight: 1.65 }}>
                  Camera and microphone access requires HTTPS when deployed.
                </p>
                {sesErr && (
                  <div role="alert" className="a-err">
                    <AlertTriangle size={15} style={{ flexShrink: 0 }} />{sesErr}
                  </div>
                )}
                <div className="ses-btn-group">
                  {!sesOn ? (
                    <button disabled={sesBusy} onClick={startSession} className="ses-btn-start">
                      <Camera size={17} />
                      {sesBusy ? "Starting…" : "Start session"}
                    </button>
                  ) : (
                    <div className="ses-btn-row">
                      <button disabled={sesBusy} onClick={scanAgain} className="ses-btn-scan">
                        <ScanLine size={16} />{sesBusy ? "Scanning…" : "Scan again"}
                      </button>
                      <button onClick={stopSession} className="ses-btn-stop">
                        <StopCircle size={16} />Stop
                      </button>
                    </div>
                  )}
                </div>
                {/* Tips */}
                <div style={{ borderRadius: 18, padding: "18px 20px", background: "rgba(13,21,18,0.03)", border: "1px solid var(--border-soft)" }}>
                  <p style={{ fontSize: "0.63rem", fontWeight: 600, letterSpacing: "0.13em", textTransform: "uppercase", color: "var(--ink-4)", marginBottom: 14, fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif" }}>Tips for best results</p>
                  <ul style={{ listStyle: "none", display: "grid", gap: 10 }}>
                    {["Face the camera in good light", "Speak at a natural pace", "Minimise background noise", "Stay still during the scan"].map(tip => (
                      <li key={tip} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: "0.85rem", color: "var(--ink-3)", fontWeight: 500 }}>
                        <CheckCircle size={13} style={{ color: "var(--green)", flexShrink: 0 }} />{tip}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Video */}
              <div className="vid-wrap" style={{ minHeight: 300 }}>
                <video ref={vidRef} className="vid-el" muted playsInline />
                {!sesOn && (
                  <div className="vid-ph">
                    <div className="vid-ph-inner">
                      <div className="vid-ph-icon"><Camera size={26} strokeWidth={1.5} /></div>
                      <span style={{ fontSize: "0.88rem", fontWeight: 700, color: "rgba(255,255,255,0.5)", fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif" }}>Camera preview</span>
                      <span style={{ fontSize: "0.77rem", color: "rgba(255,255,255,0.28)" }}>Starts with session</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Result */}
            <div className="pcard anim-fade-up" style={{ animationDelay: "0.16s" }}>
              <p className="p-eye">{sesBusy ? "Scanning now" : "Latest result"}</p>
              {sesBusy ? (
                <div className="scan-prog" style={{ marginTop: 12 }}>
                  <div className="pulse-ring" />
                  <div>
                    <p style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--ink)", fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif", marginBottom: 4 }}>Scanning new sample</p>
                    <p style={{ fontSize: "0.86rem", color: "var(--ink-3)", marginBottom: 6 }}>{Math.round(SCAN_TOTAL_MS / 1000)}-second capture in progress…</p>
                    {scanAt && <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--ink-4)" }}>Started {new Date(scanAt).toLocaleTimeString()}</span>}
                  </div>
                </div>
              ) : latest ? (
                <>
                  <h2 className="p-title" style={{ marginTop: 6 }}>{stressMeta(latest.stress_level).title}</h2>
                  <p className="p-sub" style={{ marginBottom: 20 }}>{stressMeta(latest.stress_level).summary}</p>
                  <div className="two-col" style={{ marginBottom: 20 }}>
                    <div className="sig"><p className="sig-lbl">Face signal</p><p className="sig-val">{cap(latest.face_emotion)}</p><p className="sig-meta">{pct(latest.face_confidence)} confidence</p></div>
                    <div className="sig"><p className="sig-lbl">Voice signal</p><p className="sig-val">{cap(latest.voice_emotion)}</p><p className="sig-meta">{pct(latest.voice_confidence)} confidence</p></div>
                  </div>
                  <div className="guide"><p className="guide-lbl">What this means</p><p className="guide-txt">{stressMeta(latest.stress_level).guidance}</p></div>
                </>
              ) : (
                <div className="empty" style={{ padding: "36px 16px" }}>
                  <Brain size={32} style={{ color: "var(--ink-4)", opacity: .4 }} />
                  <p>Your first reading will appear here after a short scan.</p>
                </div>
              )}
            </div>
          </Pane>
        )}

        {/* ══════════════ HISTORY ══════════════ */}
        {view === "history" && (
          <Pane>
            <div className="hero-banner" style={{ marginBottom: 20 }}>
              <div className="hero-inner">
                <p className="hero-eyebrow">Records</p>
                <h1 className="hero-title">Reading history</h1>
                <p className="hero-sub">All your saved stress check-ins — {all.length} readings total.</p>
                {all.length > 0 && (
                  <div style={{ display: "flex", gap: 8, marginTop: 20, flexWrap: "wrap" }}>
                    {(["low", "medium", "high"] as StressLevel[]).map(l => (
                      <span key={l} className={cn("pill", pillCls[l])}>
                        <span className="pill-dot" />{cap(l)}: {data.stress_distribution[l]}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="pcard anim-fade-up" style={{ animationDelay: "0.08s" }}>
              <div className="hist-filters" style={{ marginBottom: 20 }}>
                <div className="hist-filter-top">
                  <p style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--ink-4)" }}>
                    {filtered.length} result{filtered.length !== 1 ? "s" : ""}
                  </p>
                </div>
                <div className="hist-filter-row">
                  <div className="hist-search">
                    <Search size={14} style={{ position: "absolute", left: 13, top: "50%", transform: "translateY(-50%)", color: "var(--ink-4)", pointerEvents: "none", zIndex: 1 }} />
                    <Input placeholder="Search readings…" value={query} onChange={e => setQuery(e.target.value)}
                      className="pl-9 h-10 text-sm" style={{ borderRadius: 14, width: "100%" }} />
                  </div>
                  <div style={{ minWidth: 0, overflow: "hidden" }}>
                    <Select value={riskF} onValueChange={v => setRiskF(v as typeof riskF)}>
                      <SelectTrigger className="h-10 text-sm gap-1.5" style={{ borderRadius: 14, width: "100%", minWidth: 0 }}>
                        <Filter size={13} style={{ color: "var(--ink-4)", flexShrink: 0 }} /><SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All levels</SelectItem>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="medium">Medium</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div style={{ display: "grid", gap: 10 }}>
                {filtered.length === 0 ? (
                  <div className="empty">
                    <Search size={30} style={{ color: "var(--ink-4)", opacity: .38 }} />
                    <p>Nothing matches. Try adjusting the search or filter.</p>
                  </div>
                ) : filtered.map((r, i) => (
                  <div key={r.id} className="hist anim-fade-up" style={{ animationDelay: `${i * 0.025}s` }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8, flexWrap: "wrap" }}>
                        <span className={cn("pill", pillCls[r.stress_level])}><span className="pill-dot" />{r.stress_level}</span>
                        <span style={{ fontSize: "0.9rem", fontWeight: 700, color: "var(--ink)", fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif", letterSpacing: "-0.03em" }}>
                          {stressMeta(r.stress_level).title}
                        </span>
                      </div>
                      <p style={{ fontSize: "0.8rem", color: "var(--ink-3)", marginBottom: 4 }}>
                        Face: <strong style={{ color: "var(--ink-2)", fontWeight: 600 }}>{cap(r.face_emotion)}</strong>
                        {" · "}Voice: <strong style={{ color: "var(--ink-2)", fontWeight: 600 }}>{cap(r.voice_emotion)}</strong>
                      </p>
                      <p style={{ fontSize: "0.77rem", color: "var(--ink-4)" }}>
                        {new Date(r.timestamp).toLocaleString()} · {r.source.replaceAll("_", " ")}
                      </p>
                    </div>
                    <div className="hist-chips">
                      <span className="chip">Face {pct(r.face_confidence)}</span>
                      <span className="chip">Voice {pct(r.voice_confidence)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Pane>
        )}

        {/* ══════════════ PROFILE ══════════════ */}
        {view === "profile" && (
          <Pane>
            <div className="hero-banner" style={{ marginBottom: 20, alignItems: "center" }}>
              <div className="hero-inner">
                <p className="hero-eyebrow">Account</p>
                <h1 className="hero-title">{user.name}</h1>
                <p className="hero-sub">@{user.username} · {user.email}</p>
              </div>
              <div style={{
                width: 72, height: 72, borderRadius: 22, flexShrink: 0,
                background: "rgba(48,196,154,0.12)", border: "1px solid rgba(48,196,154,0.24)",
                display: "grid", placeItems: "center", color: "var(--green-hi)",
              }}>
                <User2 size={32} strokeWidth={1.6} />
              </div>
            </div>

            {/* Account details */}
            <div className="prof-card anim-fade-up" style={{ animationDelay: "0.08s" }}>
              <div className="prof-card-hd">
                <div className="prof-icon" style={{ background: "linear-gradient(135deg,var(--green-tint),var(--green-light))", color: "var(--green)", boxShadow: "0 4px 16px rgba(25,121,95,0.14)" }}>
                  <User2 size={22} />
                </div>
                <div>
                  <p className="prof-lbl-section">Account</p>
                  <h2 className="prof-card-title">Account details</h2>
                  <p className="prof-card-sub">Update your display name, username and email address.</p>
                </div>
              </div>
              <form className="prof-form" onSubmit={async e => {
                e.preventDefault(); setPErr(null); setPMsg(null);
                try { const u = await updateProfile(token, { name: pName, username: pUser, email: pEmail }); onUserUpdated(u); setPMsg("Profile updated successfully."); }
                catch (e) { setPErr(e instanceof Error ? e.message : "Could not update profile."); }
              }}>
                <div className="prof-field"><Label htmlFor="p-name" className="prof-label">Full name</Label><Input id="p-name" value={pName} onChange={e => setPName(e.target.value)} required className="prof-input" placeholder="Your full name" /></div>
                <div className="prof-field"><Label htmlFor="p-user" className="prof-label">Username</Label><Input id="p-user" value={pUser} onChange={e => setPUser(e.target.value)} minLength={3} pattern="[a-zA-Z0-9_]+" required className="prof-input" placeholder="your_username" /></div>
                <div className="prof-field"><Label htmlFor="p-email" className="prof-label">Email address</Label><Input id="p-email" type="email" value={pEmail} onChange={e => setPEmail(e.target.value)} required className="prof-input" placeholder="you@example.com" /></div>
                {pErr && <div className="a-err"><AlertTriangle size={14} style={{ flexShrink: 0 }} />{pErr}</div>}
                {pMsg && <div className="a-ok"><CheckCircle size={14} style={{ flexShrink: 0 }} />{pMsg}</div>}
                <Button type="submit" className="btn-g gap-2 w-full" style={{ height: "3.1rem" }}><Save size={15} />Save changes</Button>
              </form>
            </div>

            {/* Password */}
            <div className="prof-card anim-fade-up" style={{ animationDelay: "0.16s" }}>
              <div className="prof-card-hd">
                <div className="prof-icon" style={{ background: "linear-gradient(135deg,var(--amber-tint),var(--amber-light))", color: "var(--amber)", boxShadow: "0 4px 16px rgba(192,120,24,0.14)" }}>
                  <KeyRound size={22} />
                </div>
                <div>
                  <p className="prof-lbl-section">Security</p>
                  <h2 className="prof-card-title">Change password</h2>
                  <p className="prof-card-sub">Use a strong password of at least 8 characters.</p>
                </div>
              </div>
              <form className="prof-form" onSubmit={async e => {
                e.preventDefault(); setPwErr(null); setPwMsg(null);
                try { await changePassword(token, { current_password: curPwd, new_password: newPwd }); setCurPwd(""); setNewPwd(""); setPwMsg("Password changed. Signing out shortly…"); window.setTimeout(onLogout, 1400); }
                catch (e) { setPwErr(e instanceof Error ? e.message : "Could not change password."); }
              }}>
                <div className="prof-field">
                  <Label htmlFor="cur-pwd" className="prof-label">Current password</Label>
                  <div className="pw-wrap">
                    <Input id="cur-pwd" type={showC ? "text" : "password"} value={curPwd} onChange={e => setCurPwd(e.target.value)} minLength={8} required className="prof-input" style={{ paddingRight: "3.2rem" }} placeholder="Current password" />
                    <button type="button" onClick={() => setShowC(v => !v)} className="pw-eye">{showC ? <EyeOff size={16} /> : <Eye size={16} />}</button>
                  </div>
                </div>
                <div className="prof-field">
                  <Label htmlFor="new-pwd" className="prof-label">New password</Label>
                  <div className="pw-wrap">
                    <Input id="new-pwd" type={showN ? "text" : "password"} value={newPwd} onChange={e => setNewPwd(e.target.value)} minLength={8} required className="prof-input" style={{ paddingRight: "3.2rem" }} placeholder="Min. 8 characters" />
                    <button type="button" onClick={() => setShowN(v => !v)} className="pw-eye">{showN ? <EyeOff size={16} /> : <Eye size={16} />}</button>
                  </div>
                </div>
                {pwErr && <div className="a-err"><AlertTriangle size={14} style={{ flexShrink: 0 }} />{pwErr}</div>}
                {pwMsg && <div className="a-ok"><CheckCircle size={14} style={{ flexShrink: 0 }} />{pwMsg}</div>}
                <Button type="submit" className="btn-g gap-2 w-full" style={{ height: "3.1rem" }}><KeyRound size={15} />Update password</Button>
              </form>
            </div>

            {/* Logout */}
            <div className="prof-card anim-fade-up" style={{ animationDelay: "0.24s" }}>
              <div className="prof-card-hd">
                <div className="prof-icon" style={{ background: "linear-gradient(135deg,var(--red-tint),var(--red-light))", color: "var(--red)", boxShadow: "0 4px 16px rgba(196,60,52,0.14)" }}>
                  <LogOut size={22} />
                </div>
                <div>
                  <p className="prof-lbl-section">Session</p>
                  <h2 className="prof-card-title">Log out</h2>
                  <p className="prof-card-sub">Sign out of your workspace on this device.</p>
                </div>
              </div>
              <div style={{ padding: "0 28px 28px" }}>
                <Button variant="outline" onClick={onLogout} className="gap-2 w-full"
                  style={{ borderColor: "var(--red)", color: "var(--red)", borderRadius: 16, height: "3.1rem", fontWeight: 700, fontFamily: "var(--font-heading),'Space Grotesk',system-ui,sans-serif", letterSpacing: "-0.02em" }}>
                  <LogOut size={15} />Log out of MindPulse
                </Button>
              </div>
            </div>
          </Pane>
        )}
      </main>

      {/* ─── MOBILE NAV ─── */}
      <nav className="mob-nav" aria-label="Navigation">
        {navItems.map(tab => (
          <button key={tab.key} type="button"
            className={cn("mob-btn", view === tab.key && "on")}
            onClick={() => setView(tab.key)}>
            <span className="mob-icon">{tab.icon}</span>
            <span className="mob-lbl">{tab.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
