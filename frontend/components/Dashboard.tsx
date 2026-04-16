"use client";

import type { ReactNode } from "react";
import { useMemo, useRef, useState } from "react";
import { analyzeSample, changePassword, updateProfile } from "../lib/api";
import { captureVideoFrame, recordAudioSample } from "../lib/capture";
import type { DashboardSummary, StressLevel, User } from "../lib/types";

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

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
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
  const latestStress = data.latest_result?.stress_level ?? "low";
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
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const filteredResults = useMemo(() => {
    return historyResults.filter((result) => {
      const matchesRisk = riskFilter === "all" || result.stress_level === riskFilter;
      const searchable = `${result.face_emotion} ${result.voice_emotion} ${result.source}`.toLowerCase();
      return matchesRisk && searchable.includes(query.trim().toLowerCase());
    });
  }, [historyResults, query, riskFilter]);
  const mostCommonRisk = (Object.entries(data.stress_distribution) as [StressLevel, number][])
    .sort((a, b) => b[1] - a[1])[0]?.[0] ?? "low";

  async function captureAndAnalyze(stream: MediaStream) {
    if (!videoRef.current) {
      throw new Error("Camera preview is not ready yet.");
    }
    setSessionBusy(true);
    setSessionMessage("Capturing a short sample...");
    try {
      const audioPromise = recordAudioSample(stream, 3000);
      const faceImage = await captureVideoFrame(videoRef.current);
      const audioFile = await audioPromise;
      setSessionMessage("Analyzing sample...");
      await analyzeSample(token, { faceImage, audioFile });
      await onRefresh();
      setSessionMessage("Reading saved. You can scan again or stop the session.");
    } finally {
      setSessionBusy(false);
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
              <button className="primary-button" onClick={startBrowserSession} type="button">
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
    <main className="dashboard-shell">
      <nav className="topbar">
        <div>
          <span className="brand-mark">M</span>
          <span className="brand-text">MindPulse</span>
        </div>
        <span className="welcome-text">Welcome, {user.username}</span>
        <div className="topbar-actions">
          <button className="ghost-button" disabled={refreshing} onClick={onRefresh}>
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
          <button className="outline-button" onClick={onLogout}>Logout</button>
        </div>
      </nav>

      <section className="dashboard-hero-card">
        <div>
          <p className="eyebrow">Overview</p>
          <h1>Session summary</h1>
          <p>
            Latest readings, confidence levels, and recent stress activity for this workspace.
          </p>
        </div>
        <div className={`stress-orb ${stressTone[latestStress]}`}>
          <span>{data.latest_result ? latestStress : "No data"}</span>
          <small>{data.latest_result ? "current risk" : "current session"}</small>
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

      {activeView === "overview" && (
        <section className="metric-grid">
          <MetricCard label="Total readings" value={data.total_results.toString()} caption="Captured sessions" />
          <MetricCard label="Expression confidence" value={percent(data.average_face_confidence)} caption="Average confidence" />
          <MetricCard label="Voice confidence" value={percent(data.average_voice_confidence)} caption="Average confidence" />
          <MetricCard label="Primary level" value={mostCommonRisk} caption="Most frequent risk" />
        </section>
      )}

      {activeView === "session" ? (
        <>
          {renderSessionPanel("mobile-session-panel")}
        </>
      ) : activeView === "overview" ? (
        <>
          <section className="content-grid">
            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Distribution</p>
                  <h2>Risk level balance</h2>
                </div>
              </div>
              {(["low", "medium", "high"] as StressLevel[]).map((level) => {
                const count = data.stress_distribution[level];
                const width = Math.max((count / distributionTotal) * 100, count > 0 ? 8 : 0);
                return (
                  <div className="distribution-row" key={level}>
                    <div>
                      <span className={`status-pill ${stressTone[level]}`}>{level}</span>
                      <strong>{count}</strong>
                    </div>
                    <div className="bar-track">
                      <span className={`bar-fill ${stressTone[level]}`} style={{ width: `${width}%` }} />
                    </div>
                  </div>
                );
              })}
            </article>

            <article className="panel latest-panel">
              <p className="eyebrow">Latest reading</p>
              {data.latest_result ? (
                <>
                  <h2>{data.latest_result.face_emotion} expression, {data.latest_result.voice_emotion} voice</h2>
                  <p>
                    Last captured through <strong>{formatSource(data.latest_result.source)}</strong> at{" "}
                    {new Date(data.latest_result.timestamp).toLocaleString()}.
                  </p>
                  <div className="confidence-pair">
                    <span>Face {percent(data.latest_result.face_confidence)}</span>
                    <span>Voice {percent(data.latest_result.voice_confidence)}</span>
                  </div>
                </>
              ) : (
                <p>No readings yet. Start a capture session to populate your workspace.</p>
              )}
            </article>
          </section>

          <section className="panel trend-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Trend</p>
                <h2>Recent risk timeline</h2>
              </div>
            </div>
            <div className="timeline-strip">
              {data.recent_results.slice(0, 12).reverse().map((result) => (
                <div className="timeline-item" key={result.id}>
                  <span className={`timeline-dot ${stressTone[result.stress_level]}`} />
                  <small>{new Date(result.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
                </div>
              ))}
              {data.recent_results.length === 0 && <p>No timeline data yet.</p>}
            </div>
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
                <h3>No matching readings</h3>
                <p>Try changing the search term or risk filter.</p>
              </div>
            ) : (
              filteredResults.map((result) => (
                <article className="history-card" key={result.id}>
                  <div>
                    <span className={`status-pill ${stressTone[result.stress_level]}`}>{result.stress_level}</span>
                    <h3>{result.face_emotion} expression, {result.voice_emotion} voice</h3>
                    <p>{new Date(result.timestamp).toLocaleString()} - {formatSource(result.source)}</p>
                  </div>
                  <div className="history-confidence">
                    <span>Expression {percent(result.face_confidence)}</span>
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
                  <td colSpan={5}>No results yet.</td>
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
