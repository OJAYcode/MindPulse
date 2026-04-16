"use client";

import { FormEvent, useState } from "react";
import { loginUser, registerUser } from "../lib/api";
import type { User } from "../lib/types";

type AuthPanelProps = {
  onAuthenticated: (token: string, user: User) => void;
};

export function AuthPanel({ onAuthenticated }: AuthPanelProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const response =
        mode === "register"
          ? await registerUser({ name, username, email, password })
          : await loginUser({ email, password });
      onAuthenticated(response.token, response.user);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Authentication failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-hero">
        <div className="eyebrow">MindPulse</div>
        <h1>Stress monitoring for live sessions.</h1>
        <p>
          Review readings, spot elevated patterns, and keep session history in one secure workspace.
        </p>
        <div className="hero-note">
          <span>For check-ins, reviews, and session monitoring.</span>
        </div>
      </section>

      <section className="auth-card">
        <div className="segmented-control" aria-label="Authentication mode">
          <button className={mode === "login" ? "active" : ""} onClick={() => setMode("login")} type="button">
            Sign in
          </button>
          <button className={mode === "register" ? "active" : ""} onClick={() => setMode("register")} type="button">
            Create account
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <h2>{mode === "login" ? "Sign in" : "Create account"}</h2>
          <p className="form-copy">
            {mode === "login"
              ? "Access your MindPulse workspace."
              : "Set up access for this device."}
          </p>

          {mode === "register" && (
            <>
              <label>
                Full name
                <input value={name} onChange={(event) => setName(event.target.value)} minLength={2} required />
              </label>
              <label>
                Username
                <input
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  minLength={3}
                  pattern="[a-zA-Z0-9_]+"
                  required
                />
              </label>
            </>
          )}

          <label>
            Email address
            <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} required />
          </label>

          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={8}
              required
            />
          </label>

          {error && <div className="form-error">{error}</div>}

          <button className="primary-button" disabled={submitting} type="submit">
            {submitting ? "Please wait..." : mode === "login" ? "Log in" : "Sign up"}
          </button>
        </form>
      </section>
    </main>
  );
}
