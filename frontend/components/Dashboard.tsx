"use client";

import type { ReactNode } from "react";
import { useMemo, useRef, useState } from "react";
import { analyzeSample, changePassword, updateProfile } from "../lib/api";
import { captureVideoFrames, delay, recordAudioSample, waitForVideoReady } from "../lib/capture";
import type { DashboardSummary, InferenceResult, StressLevel, User } from "../lib/types";

type DashboardProps = {
  user: User;
  token: string;
  summary: DashboardSummary | null;
  onRefresh: () => Promise<void>;
  refreshing: boolean;
  onUserUpdated: (user: User) => void;
  onLogout: () => void;
};

type DashboardView = "overview" | "session" | "history" | "profile";

const stressTone: Record<StressLevel, string> = {
  low: "tone-low",
  medium: "tone-medium",
  high: "tone-high"
};

const desktopTabs: { key: DashboardView; label: string; icon: ReactNode }[] = [
  {
    key: "overview",
    label: "Overview",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path
          d="M4 12.5 12 5l8 7.5M7.5 10v8h9v-8"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.9"
        />
      </svg>
    )
  },
  {
    key: "session",
    label: "Session",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path
          d="M7 9a2 2 0 0 1 2-2h5l3 3v5a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V9Zm8 0v2h2"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.9"
        />
      </svg>
    )
  },
  {
    key: "history",
    label: "History",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path
          d="M12 7.5v5l3.5 2M20 12a8 8 0 1 1-2.34-5.66M20 4v4h-4"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.9"
        />
      </svg>
    )
  },
  {
    key: "profile",
    label: "Profile",
    icon: (
      <svg aria-hidden="true" viewBox="0 0 24 24">
        <path
          d="M12 12a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm-6 7a6 6 0 0 1 12 0"
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="1.9"
        />
      </svg>
    )
  }
];

const mobileTabs = desktopTabs;
const SCAN_READY_DELAY_MS = 900;
const SCAN_RECORDING_MS = 5500;
const SCAN_FRAME_COUNT = 5;
const SCAN_FRAME_INTERVAL_MS = 850;
const SCAN_TOTAL_MS = SCAN_READY_DELAY_MS + SCAN_RECORDING_MS;

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function titleCase(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function emotionCopy(label: string) {
  const key = label.toLowerCase();
  const map: Record<string, string> = {
    calm: "sounded calm",
    stressed: "sounded tense",
    neutral: "looked steady",
    happy: "looked positive",
    sad: "looked low",
    angry: "looked tense"
  };
  return map[key] ?? label;
}

function stressCopy(level: StressLevel) {
  if (level === "high") {
    return {
      title: "Needs a pause",
      badge: "High",
      summary: "This check-in suggests you may be feeling quite strained right now.",
      guidance: "Take a short pause, slow your breathing, and check in again after a moment.",
      feeling: "You may be under quite a bit of pressure right now.",
      nextStepTitle: "Best next step"
    };
  }
  if (level === "medium") {
    return {
      title: "A bit tense",
      badge: "Medium",
      summary: "This check-in suggests some tension or pressure at the moment.",
      guidance: "A short break or a calmer environment may help before the next check-in.",
      feeling: "There are some signs of tension, but not an extreme level.",
      nextStepTitle: "Helpful next step"
    };
  }
  return {
    title: "Steady",
    badge: "Low",
    summary: "This check-in suggests a calmer or more stable state.",
    guidance: "You can keep going and scan again later if you want another update.",
    feeling: "You seem fairly steady at the moment.",
    nextStepTitle: "What to do next"
  };
}

function resultHeadline(result: DashboardSummary["latest_result"]) {
  if (!result) {
    return "Ready for your first check-in";
  }
  if (result.stress_level === "high") {
    return "You may need a short pause";
  }
  if (result.stress_level === "medium") {
    return "You seem a bit tense right now";
  }
  return "You seem fairly steady right now";
}

function faceObservation(label: string) {
  const key = label.toLowerCase();
  const map: Record<string, string> = {
    calm: "Your face looked calm and settled.",
    neutral: "Your face looked fairly steady.",
    happy: "Your face showed some positive energy.",
    sad: "Your face showed signs of low mood or tiredness.",
    angry: "Your face showed visible tension.",
    stressed: "Your face showed visible stress."
  };
  return map[key] ?? `Your face signal was read as ${label}.`;
}

function voiceObservation(label: string) {
  const key = label.toLowerCase();
  const map: Record<string, string> = {
    calm: "Your voice sounded calm.",
    neutral: "Your voice sounded steady.",
    happy: "Your voice sounded more upbeat.",
    sad: "Your voice sounded low or subdued.",
    angry: "Your voice sounded tense.",
    stressed: "Your voice sounded tense or pressured."
  };
  return map[key] ?? `Your voice signal was read as ${label}.`;
}

function emptySummary(): DashboardSummary {
  return {
    total_results: 0,
    latest_result: null,
    stress_distribution: { low: 0, medium: 0, high: 0 },
    average_face_confidence: 0,
    average_voice_confidence: 0,
    high_stress_rate: 0,
    recent_results: []
  };
}

function isDemoRecord(source: string) {
  return source === "demo_session";
}

function visibleSummary(summary: DashboardSummary, userId: number): DashboardSummary {
  const activeResults = summary.recent_results.filter((result) => {
    return !isDemoRecord(result.source) && (result.user_id === userId || result.user_id === null || result.user_id === undefined);
  });
  const stressDistribution = activeResults.reduce(
    (distribution, result) => {
      distribution[result.stress_level] += 1;
      return distribution;
    },
    { low: 0, medium: 0, high: 0 } as Record<StressLevel, number>
  );
  const total = activeResults.length;
  const averageFace = activeResults.reduce((sum, result) => sum + result.face_confidence, 0) / Math.max(total, 1);
  const averageVoice = activeResults.reduce((sum, result) => sum + result.voice_confidence, 0) / Math.max(total, 1);
  return {
    ...summary,
    total_results: total,
    latest_result: activeResults[0] ?? null,
    stress_distribution: stressDistribution,
    average_face_confidence: total ? averageFace : 0,
    average_voice_confidence: total ? averageVoice : 0,
    high_stress_rate: total ? stressDistribution.high / total : 0,
    recent_results: activeResults
  };
}

export function Dashboard({ user, token, summary, onRefresh, refreshing, onUserUpdated, onLogout }: DashboardProps) {
  const rawData = summary ?? emptySummary();
  const data = visibleSummary(rawData, user.id);
  const historyResults = rawData.recent_results;
  const distributionTotal = Math.max(data.total_results, 1);
  const [activeView, setActiveView] = useState<DashboardView>("overview");
  const [riskFilter, setRiskFilter] = useState<"all" | StressLevel>("all");
  const [query, setQuery] = useState("");
  const [profileName, setProfileName] = useState(user.name);
  const [profileUsername, setProfileUsername] = useState(user.username);
  const [profileEmail, setProfileEmail] = useState(user.email);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [profileMessage, setProfileMessage] = useState<string | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [sessionMessage, setSessionMessage] = useState("Camera and microphone are ready when you start a session.");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [currentScanResult, setCurrentScanResult] = useState<InferenceResult | null>(null);
  const [scanStartedAt, setScanStartedAt] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanRunRef = useRef(0);
  const filteredResults = useMemo(() => {
    return historyResults.filter((result) => {
      const matchesRisk = riskFilter === "all" || result.stress_level === riskFilter;
      const searchable = `${result.face_emotion} ${result.voice_emotion} ${result.source}`.toLowerCase();
      return matchesRisk && searchable.includes(query.trim().toLowerCase());
    });
  }, [historyResults, query, riskFilter]);
  const mostCommonRisk = (Object.entries(data.stress_distribution) as [StressLevel, number][])
    .sort((a, b) => b[1] - a[1])[0]?.[0] ?? "low";
  const latestResult = currentScanResult ?? data.latest_result;
  const latestStress = latestResult?.stress_level ?? "low";
  const latestStressCopy = stressCopy(latestStress);

  async function captureAndAnalyze(stream: MediaStream) {
    if (!videoRef.current) {
      throw new Error("Camera preview is not ready yet.");
    }
    const scanRunId = scanRunRef.current + 1;
    scanRunRef.current = scanRunId;
    setCurrentScanResult(null);
    setScanStartedAt(new Date().toISOString());
    setSessionBusy(true);
    setSessionMessage("Get ready. Keep your face visible and speak naturally.");
    try {
      await waitForVideoReady(videoRef.current);
      if (scanRunRef.current !== scanRunId) return;
      await delay(SCAN_READY_DELAY_MS);
      if (scanRunRef.current !== scanRunId) return;
      setSessionMessage("Recording a short sample. Please keep speaking until it finishes.");
      const audioPromise = recordAudioSample(stream, SCAN_RECORDING_MS);
      const faceImages = await captureVideoFrames(videoRef.current, SCAN_FRAME_COUNT, SCAN_FRAME_INTERVAL_MS);
      await delay(Math.max(SCAN_RECORDING_MS - SCAN_FRAME_COUNT * SCAN_FRAME_INTERVAL_MS, 0));
      if (scanRunRef.current !== scanRunId) return;
      const audioFile = await audioPromise;
      if (scanRunRef.current !== scanRunId) return;
      setSessionMessage("Analyzing sample...");
      const result = await analyzeSample(token, { faceImages, audioFile });
      if (scanRunRef.current !== scanRunId) return;
      setCurrentScanResult(result);
      await onRefresh();
      setSessionMessage("New reading saved. You can scan again or stop the session.");
    } finally {
      if (scanRunRef.current === scanRunId) {
        setSessionBusy(false);
      }
    }
  }

  async function startBrowserSession() {
    setSessionError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: true
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        await waitForVideoReady(videoRef.current);
      }
      setSessionActive(true);
      setSessionMessage("Session started. Keep your face visible and speak naturally.");
      await captureAndAnalyze(stream);
    } catch (error) {
      setSessionError(error instanceof Error ? error.message : "Could not start session.");
      stopBrowserSession();
    }
  }

  function stopBrowserSession() {
    scanRunRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setSessionActive(false);
    setSessionBusy(false);
    setSessionMessage("Session stopped.");
  }

  async function scanAgain() {
    if (!streamRef.current || sessionBusy) return;
    setSessionError(null);
    try {
      await captureAndAnalyze(streamRef.current);
    } catch (error) {
      setSessionError(error instanceof Error ? error.message : "Could not analyze sample.");
    }
  }

  function renderSessionPanel(className = "") {
    return (
      <section className={`panel session-panel ${className}`.trim()}>
        <div className="session-copy">
          <p className="eyebrow">Session</p>
          <h2>{sessionActive ? "Session running" : "Start a scan"}</h2>
          <p>{sessionMessage}</p>
          <p className="session-note">Camera and microphone access requires HTTPS when deployed.</p>
          {sessionError && <div className="form-error">{sessionError}</div>}
          <div className="session-actions">
            {!sessionActive ? (
              <button className="primary-button" disabled={sessionBusy} onClick={startBrowserSession} type="button">
                Start session
              </button>
            ) : (
              <>
                <button className="primary-button" disabled={sessionBusy} onClick={scanAgain} type="button">
                  {sessionBusy ? "Scanning..." : "Scan again"}
                </button>
                <button className="outline-button" onClick={stopBrowserSession} type="button">
                  Stop session
                </button>
              </>
            )}
          </div>
        </div>
        <div className="video-preview-wrap">
          <video ref={videoRef} className="video-preview" muted playsInline />
          {!sessionActive && <span>Camera preview</span>}
        </div>
      </section>
    );
  }

  return (
    <main className={`dashboard-shell dashboard-view-${activeView}`}>
      <nav className="topbar">
        <div className="topbar-brand">
          <div>
            <span className="brand-mark">M</span>
            <span className="brand-text">MindPulse</span>
          </div>
          <span className="welcome-text">Welcome, {user.username}</span>
        </div>
        <div className="topbar-actions">
          <button className="ghost-button" disabled={refreshing} onClick={onRefresh}>
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
          <button className="outline-button" onClick={onLogout}>Logout</button>
        </div>
        <button
          aria-label="Refresh dashboard"
          className="mobile-refresh-button"
          disabled={refreshing}
          onClick={onRefresh}
          type="button"
        >
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path
              d="M20 4v5h-5M4 20v-5h5M6.6 9.2A7 7 0 0 1 18 6.5L20 9M4 15l2-2.5A7 7 0 0 0 17.4 14.8"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="1.9"
            />
          </svg>
        </button>
      </nav>

      <section className="dashboard-hero-card">
        <div>
          <p className="eyebrow">Overview</p>
          <h1>{resultHeadline(latestResult)}</h1>
          <p>
            {latestResult
              ? latestStressCopy.feeling
              : "MindPulse is ready to read a short face and voice sample when you begin."}
          </p>
        </div>
        <div className={`stress-orb ${stressTone[latestStress]}`}>
          <span>{data.latest_result ? latestStressCopy.badge : "Ready"}</span>
          <small>{data.latest_result ? "current state" : "waiting to scan"}</small>
        </div>
      </section>

      <section className="dashboard-tabs desktop-tabs" aria-label="Dashboard sections">
        {desktopTabs.filter((tab) => tab.key !== "session").map((tab) => (
          <button
            key={tab.key}
            className={activeView === tab.key ? "active" : ""}
            onClick={() => setActiveView(tab.key)}
            type="button"
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </section>

      {activeView === "overview" && renderSessionPanel("desktop-session-panel")}

      {activeView === "session" ? (
        <>
          {renderSessionPanel("mobile-session-panel")}
          <section className="panel result-panel">
            <p className="eyebrow">{sessionBusy ? "Current scan" : "Latest result"}</p>
            {sessionBusy ? (
              <div className="scan-progress-card">
                <span className="pulse-dot" />
                <div>
                  <h2>Scanning new sample</h2>
                  <p>
                    This is a fresh {Math.round(SCAN_TOTAL_MS / 1000)} second scan. The previous result is hidden until this one finishes.
                  </p>
                  {scanStartedAt && <small>Started at {new Date(scanStartedAt).toLocaleTimeString()}</small>}
                </div>
              </div>
            ) : latestResult ? (
              <>
                <h2>{stressCopy(latestResult.stress_level).title}</h2>
                <p>{stressCopy(latestResult.stress_level).summary}</p>
                <div className="result-signal-grid">
                  <div className="result-signal-card">
                    <span>Face signal</span>
                    <strong>{titleCase(latestResult.face_emotion)}</strong>
                    <small>{emotionCopy(latestResult.face_emotion)} • {percent(latestResult.face_confidence)}</small>
                  </div>
                  <div className="result-signal-card">
                    <span>Voice signal</span>
                    <strong>{titleCase(latestResult.voice_emotion)}</strong>
                    <small>{emotionCopy(latestResult.voice_emotion)} • {percent(latestResult.voice_confidence)}</small>
                  </div>
                </div>
                <div className="result-guidance">
                  <strong>What this means</strong>
                  <p>{stressCopy(latestResult.stress_level).guidance}</p>
                </div>
              </>
            ) : (
              <p>Your first reading will appear here after a short camera and microphone scan.</p>
            )}
          </section>
        </>
      ) : activeView === "overview" ? (
        <>
          <section className="content-grid result-story-grid">
            <article className="panel story-panel">
              <p className="eyebrow">What this means</p>
              {latestResult ? (
                <>
                  <h2>{stressCopy(latestResult.stress_level).title}</h2>
                  <p>{stressCopy(latestResult.stress_level).summary}</p>
                  <div className="story-note">
                    <strong>{stressCopy(latestResult.stress_level).nextStepTitle}</strong>
                    <p>{stressCopy(latestResult.stress_level).guidance}</p>
                  </div>
                </>
              ) : (
                <>
                  <h2>Your first reading starts here</h2>
                  <p>Once you complete a scan, this section will describe your current state in plain language.</p>
                </>
              )}
            </article>

            <article className="panel story-panel">
              <p className="eyebrow">What we noticed</p>
              {latestResult ? (
                <>
                  <h2>Face and voice signals</h2>
                  <div className="story-list">
                    <div className="story-item">
                      <strong>Face</strong>
                      <p>{faceObservation(latestResult.face_emotion)}</p>
                    </div>
                    <div className="story-item">
                      <strong>Voice</strong>
                      <p>{voiceObservation(latestResult.voice_emotion)}</p>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <h2>Signals will appear after scanning</h2>
                  <p>The app will summarize the face and voice cues it used, so the result is easier to understand.</p>
                </>
              )}
            </article>

            <article className="panel story-panel">
              <p className="eyebrow">Quick overview</p>
              {latestResult ? (
                <>
                  <h2>Latest check-in details</h2>
                  <div className="story-list compact-story-list">
                    <div className="story-item">
                      <strong>Stress level</strong>
                      <p>{stressCopy(latestResult.stress_level).title}</p>
                    </div>
                    <div className="story-item">
                      <strong>Session count</strong>
                      <p>{data.total_results} saved {data.total_results === 1 ? "check-in" : "check-ins"} so far.</p>
                    </div>
                    <div className="story-item">
                      <strong>Saved at</strong>
                      <p>{new Date(latestResult.timestamp).toLocaleString()}</p>
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <h2>Your timeline is ready</h2>
                  <p>Completed scans will be saved here with the time, level, and source of each reading.</p>
                </>
              )}
            </article>
          </section>
        </>
      ) : activeView === "history" ? (
        <section className="panel history-panel">
          <div className="panel-header history-header">
            <div>
              <p className="eyebrow">History</p>
              <h2>Reading history</h2>
            </div>
            <div className="history-controls">
              <input
                aria-label="Search readings"
                placeholder="Search readings"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
              <select
                aria-label="Filter risk level"
                value={riskFilter}
                onChange={(event) => setRiskFilter(event.target.value as "all" | StressLevel)}
              >
                <option value="all">All levels</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>

          <div className="history-list">
            {filteredResults.length === 0 ? (
              <div className="empty-state">
                <h3>Nothing matches this filter</h3>
                <p>Adjust the search or level filter to bring readings back into view.</p>
              </div>
            ) : (
              filteredResults.map((result) => (
                <article className="history-card" key={result.id}>
                  <div>
                    <span className={`status-pill ${stressTone[result.stress_level]}`}>{result.stress_level}</span>
                    <h3>{stressCopy(result.stress_level).title}</h3>
                    <p>
                      Face signal {emotionCopy(result.face_emotion)} and voice signal {emotionCopy(result.voice_emotion)}.
                    </p>
                    <p>{new Date(result.timestamp).toLocaleString()} • {formatSource(result.source)}</p>
                  </div>
                  <div className="history-confidence">
                    <span>Face {percent(result.face_confidence)}</span>
                    <span>Voice {percent(result.voice_confidence)}</span>
                  </div>
                </article>
              ))
            )}
          </div>
        </section>
      ) : (
        <section className="profile-grid">
          <article className="panel profile-panel">
            <p className="eyebrow">Profile</p>
            <h2>Account details</h2>
            <form
              onSubmit={async (event) => {
                event.preventDefault();
                setProfileError(null);
                setProfileMessage(null);
                try {
                  const updatedUser = await updateProfile(token, {
                    name: profileName,
                    username: profileUsername,
                    email: profileEmail
                  });
                  onUserUpdated(updatedUser);
                  setProfileMessage("Profile updated.");
                } catch (error) {
                  setProfileError(error instanceof Error ? error.message : "Could not update profile.");
                }
              }}
            >
              <label>
                Full name
                <input value={profileName} onChange={(event) => setProfileName(event.target.value)} required />
              </label>
              <label>
                Username
                <input
                  value={profileUsername}
                  onChange={(event) => setProfileUsername(event.target.value)}
                  minLength={3}
                  pattern="[a-zA-Z0-9_]+"
                  required
                />
              </label>
              <label>
                Email address
                <input type="email" value={profileEmail} onChange={(event) => setProfileEmail(event.target.value)} required />
              </label>
              {profileError && <div className="form-error">{profileError}</div>}
              {profileMessage && <div className="form-success">{profileMessage}</div>}
              <button className="primary-button" type="submit">Save changes</button>
            </form>
          </article>

          <article className="panel profile-panel">
            <p className="eyebrow">Security</p>
            <h2>Change password</h2>
            <form
              onSubmit={async (event) => {
                event.preventDefault();
                setProfileError(null);
                setProfileMessage(null);
                try {
                  await changePassword(token, {
                    current_password: currentPassword,
                    new_password: newPassword
                  });
                  setCurrentPassword("");
                  setNewPassword("");
                  setProfileMessage("Password changed. Sign in again with your new password.");
                  window.setTimeout(onLogout, 1200);
                } catch (error) {
                  setProfileError(error instanceof Error ? error.message : "Could not change password.");
                }
              }}
            >
              <label>
                Current password
                <input
                  type="password"
                  value={currentPassword}
                  onChange={(event) => setCurrentPassword(event.target.value)}
                  minLength={8}
                  required
                />
              </label>
              <label>
                New password
                <input
                  type="password"
                  value={newPassword}
                  onChange={(event) => setNewPassword(event.target.value)}
                  minLength={8}
                  required
                />
              </label>
              <button className="primary-button" type="submit">Update password</button>
            </form>

            <div className="profile-logout-block">
              <p className="eyebrow">Session</p>
              <h2>Log out</h2>
              <p>Sign out of your workspace on this device when you are done.</p>
              <button className="outline-button profile-logout-button" onClick={onLogout} type="button">
                Logout
              </button>
            </div>
          </article>
        </section>
      )}

      <section className="panel table-panel compact-table">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Recent</p>
            <h2>Latest records</h2>
          </div>
        </div>
        <div className="responsive-table">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Expression</th>
                <th>Voice</th>
                <th>Risk</th>
                <th>Channel</th>
              </tr>
            </thead>
            <tbody>
              {historyResults.length === 0 ? (
                <tr>
                  <td colSpan={5}>Your recent readings will appear here after your first scan.</td>
                </tr>
              ) : (
                historyResults.map((result) => (
                  <tr key={result.id}>
                    <td data-label="Time">{new Date(result.timestamp).toLocaleString()}</td>
                    <td data-label="Expression">{result.face_emotion} - {percent(result.face_confidence)}</td>
                    <td data-label="Voice">{result.voice_emotion} - {percent(result.voice_confidence)}</td>
                    <td data-label="Risk"><span className={`status-pill ${stressTone[result.stress_level]}`}>{result.stress_level}</span></td>
                    <td data-label="Channel">{formatSource(result.source)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      <nav className="mobile-bottom-nav" aria-label="Mobile dashboard navigation">
        {mobileTabs.map((tab) => (
          <button
            key={tab.key}
            className={activeView === tab.key ? "active" : ""}
            onClick={() => setActiveView(tab.key)}
            type="button"
          >
            <span className="tab-icon">{tab.icon}</span>
            <span className="tab-label">{tab.label}</span>
          </button>
        ))}
      </nav>
    </main>
  );
}

function formatSource(source: string) {
  return source.replaceAll("_", " ");
}

function MetricCard({ label, value, caption }: { label: string; value: string; caption: string }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{caption}</small>
    </article>
  );
}
