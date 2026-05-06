import type { AuthResponse, DashboardSummary, InferenceResult, User } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {})
    }
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export function registerUser(payload: { name: string; username: string; email: string; password: string }) {
  return request<AuthResponse>("/auth/register", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function loginUser(payload: { email: string; password: string }) {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getMe(token: string) {
  return request<User>("/auth/me", {
    headers: { Authorization: `Bearer ${token}` }
  });
}

export function getDashboardSummary(token: string) {
  return request<DashboardSummary>("/dashboard/summary", {
    headers: { Authorization: `Bearer ${token}` }
  });
}

export function updateProfile(
  token: string,
  payload: { name: string; username: string; email: string }
) {
  return request<User>("/auth/profile", {
    method: "PATCH",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify(payload)
  }).catch((error) => {
    if (error instanceof Error && error.message.toLowerCase().includes("not found")) {
      return request<User>("/auth/profile", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload)
      });
    }
    throw error;
  });
}

export function changePassword(
  token: string,
  payload: { current_password: string; new_password: string }
) {
  return request<{ status: string }>("/auth/change-password", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify(payload)
  });
}

export async function analyzeSample(
  token: string,
  payload: { faceImage?: Blob; faceImages?: Blob[]; audioFile: Blob }
) {
  const formData = new FormData();
  const faceImages = payload.faceImages?.length ? payload.faceImages : payload.faceImage ? [payload.faceImage] : [];
  faceImages.forEach((faceImage, index) => {
    formData.append("face_images", faceImage, `frame-${index + 1}.jpg`);
  });
  formData.append("audio_file", payload.audioFile, "audio.wav");
  const response = await fetch(`${API_BASE_URL}/sessions/analyze-sample`, {
    method: "POST",
    cache: "no-store",
    headers: { Authorization: `Bearer ${token}` },
    body: formData
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Could not analyze sample" }));
    throw new Error(error.detail ?? "Could not analyze sample");
  }
  return response.json() as Promise<InferenceResult>;
}
