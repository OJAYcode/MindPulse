"use client";

import { useEffect, useState } from "react";
import { AuthPanel } from "@/components/AuthPanel";
import { Dashboard } from "@/components/Dashboard";
import { getDashboardSummary, getMe } from "@/lib/api";
import type { DashboardSummary, User } from "@/lib/types";

const TOKEN_KEY = "mindpulse_auth_token";

export default function Home() {
  const [token,      setToken]      = useState<string | null>(null);
  const [user,       setUser]       = useState<User | null>(null);
  const [summary,    setSummary]    = useState<DashboardSummary | null>(null);
  const [loading,    setLoading]    = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem(TOKEN_KEY);
    if (!stored) { setLoading(false); return; }
    setToken(stored);
    Promise.all([getMe(stored), getDashboardSummary(stored)])
      .then(([u, s]) => { setUser(u); setSummary(s); })
      .catch(() => { window.localStorage.removeItem(TOKEN_KEY); setToken(null); })
      .finally(() => setLoading(false));
  }, []);

  async function handleAuthenticated(t: string, u: User) {
    window.localStorage.setItem(TOKEN_KEY, t);
    setToken(t); setUser(u);
    setSummary(await getDashboardSummary(t));
  }

  async function refreshDashboard() {
    if (!token) return;
    setRefreshing(true);
    try { setSummary(await getDashboardSummary(token)); }
    finally { setRefreshing(false); }
  }

  function handleLogout() {
    window.localStorage.removeItem(TOKEN_KEY);
    setToken(null); setUser(null); setSummary(null);
  }

  if (loading) {
    return (
      <div className="loading-root">
        <div className="loading-card">
          <span className="pulse-dot" />
          Opening MindPulse…
        </div>
      </div>
    );
  }


  if (!token || !user) {
    return <AuthPanel onAuthenticated={handleAuthenticated} />;
  }

  return (
    <Dashboard
      user={user}
      token={token}
      summary={summary}
      onRefresh={refreshDashboard}
      refreshing={refreshing}
      onUserUpdated={setUser}
      onLogout={handleLogout}
    />
  );
}
