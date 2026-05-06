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
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function switchMode(nextMode: "login" | "register") {
    setMode(nextMode);
    setName("");
    setUsername("");
    setEmail("");
    setPassword("");
    setShowPassword(false);
    setError(null);
    setSubmitting(false);
  }

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
          <button className={mode === "login" ? "active" : ""} onClick={() => switchMode("login")} type="button">
            Sign in
          </button>
          <button className={mode === "register" ? "active" : ""} onClick={() => switchMode("register")} type="button">
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
            <div className="password-field">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={8}
                required
              />
              <button
                aria-label={showPassword ? "Hide password" : "Show password"}
                className="password-toggle"
                onClick={() => setShowPassword((current) => !current)}
                type="button"
              >
                {showPassword ? (
                  <svg aria-hidden="true" viewBox="0 0 24 24">
                    <path
                      d="m4 4 16 16M10.58 10.58A2 2 0 0 0 12 14a2 2 0 0 0 1.42-.58M9.88 5.09A10.94 10.94 0 0 1 12 4.9c5.2 0 8.77 3.45 10 7.1a11.5 11.5 0 0 1-3.12 4.78M6.1 6.09A11.46 11.46 0 0 0 2 12c1.23 3.65 4.8 7.1 10 7.1 1.75 0 3.31-.39 4.67-1.02"
                      fill="none"
                      stroke="currentColor"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="1.9"
                    />
                  </svg>
                ) : (
                  <svg aria-hidden="true" viewBox="0 0 24 24">
                    <path
                      d="M2 12c1.23-3.65 4.8-7.1 10-7.1S20.77 8.35 22 12c-1.23 3.65-4.8 7.1-10 7.1S3.23 15.65 2 12Zm10 2.1a2.1 2.1 0 1 0 0-4.2 2.1 2.1 0 0 0 0 4.2Z"
                      fill="none"
                      stroke="currentColor"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="1.9"
                    />
                  </svg>
                )}
              </button>
            </div>
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
