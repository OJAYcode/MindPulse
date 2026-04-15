export type StressLevel = "low" | "medium" | "high";

export type User = {
  id: number;
  name: string;
  username: string;
  email: string;
  created_at: string;
};

export type AuthResponse = {
  token: string;
  token_type: "bearer";
  user: User;
};

export type InferenceResult = {
  id: number;
  user_id?: number | null;
  timestamp: string;
  face_emotion: string;
  face_confidence: number;
  voice_emotion: string;
  voice_confidence: number;
  stress_level: StressLevel;
  source: string;
};

export type DashboardSummary = {
  total_results: number;
  latest_result: InferenceResult | null;
  stress_distribution: Record<StressLevel, number>;
  average_face_confidence: number;
  average_voice_confidence: number;
  high_stress_rate: number;
  recent_results: InferenceResult[];
};
