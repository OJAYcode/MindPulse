"use client";

import { useEffect, useState } from "react";
import { AuthPanel } from "../components/AuthPanel";
import { Dashboard } from "../components/Dashboard";
import { getDashboardSummary, getMe } from "../lib/api";
import type { DashboardSummary, User } from "../lib/types";

const TOKEN_KEY = "mindpulse_auth_token";

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    const storedToken = window.localStorage.getItem(TOKEN_KEY);
    if (!storedToken) {
      setLoading(false);
      return;
    }
    setToken(storedToken);
    Promise.all([getMe(storedToken), getDashboardSummary(storedToken)])
      .then(([currentUser, dashboardSummary]) => {
        setUser(currentUser);
        setSummary(dashboardSummary);
      })
      .catch(() => {
        window.localStorage.removeItem(TOKEN_KEY);
        setToken(null);
      })
      .finally(() => setLoading(false));
  }, []);

  async function handleAuthenticated(nextToken: string, nextUser: User) {
    window.localStorage.setItem(TOKEN_KEY, nextToken);
    setToken(nextToken);
    setUser(nextUser);
    setSummary(await getDashboardSummary(nextToken));
  }

  async function refreshDashboard() {
    if (!token) return;
    setRefreshing(true);
    try {
      setSummary(await getDashboardSummary(token));
    } finally {
      setRefreshing(false);
    }
  }

  function handleLogout() {
    window.localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
    setSummary(null);
  }

  if (loading) {
    return (
      <main className="shell center-shell">
        <div className="loading-card">
          <span className="pulse-dot" />
          Opening MindPulse
        </div>
      </main>
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
