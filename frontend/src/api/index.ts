const BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL ?? "";
const USE_MOCK = !BASE_URL;

export interface HealthResponse {
  status: "ok" | "offline";
  service?: string;
}

export interface SessionStartRequest {
  user_id: string;
  device: string;
  fps: number;
  resolution: [number, number];
}

export interface SessionStartResponse {
  session_id: string;
  ws_url: string;
}

export interface SessionSummary {
  session_id: string;
  user_id: string;
  device: string;
  fps: number;
  resolution: [number, number] | number[];
  created_at_ms: number;
  shot_count: number;
}

export interface ShotMetrics {
  release_angle: number;
  arc: number;
  tempo: number;
  drift: number;
}

export interface Shot {
  id: string;
  session_id: string;
  timestamp: string;
  result: "made" | "miss";
  coaching_cue: string;
  quality_score: number;
  metrics: ShotMetrics;
}

export interface ShotDetail extends Shot {
  mistakes: string[];
  context: string;
  clip_url: string | null;
}

export interface Drill {
  name: string;
  description: string;
  link: string;
}

export interface ReplayAnalysis {
  what_went_well: string[];
  what_to_fix: string[];
  moment_annotations: Array<{ timestamp_s: number; note: string }>;
  primary_drill: Drill;
  backup_drill: Drill;
  next_shot_focus: string;
}

interface ApiShotMetrics {
  release_angle: number | null;
  arc_height_ratio: number | null;
  shot_tempo: number | null;
  torso_drift: number | null;
}

interface ApiMistake {
  message: string;
}

interface ApiShot {
  shot_id: string;
  session_id: string;
  made: boolean;
  timestamp_ms: number;
  metrics: ApiShotMetrics;
  mistakes?: ApiMistake[];
  clip_url?: string | null;
}

interface ApiDrill {
  name: string;
  duration_min: number;
  steps: string[];
  links: string[];
}

interface ApiMoment {
  t_sec: number;
  observation: string;
  correction: string;
  tag: string;
}

interface ApiReplay {
  what_went_well: string[];
  what_to_fix: string[];
  moment_annotations?: ApiMoment[];
  drill?: ApiDrill | null;
  backup_drill?: ApiDrill | null;
  next_shot_focus?: string | null;
}

function toUiShot(api: ApiShot): Shot {
  const arcRatio = api.metrics.arc_height_ratio ?? 0;
  const arcDisplay = Math.max(0, Math.min(100, arcRatio * 100));
  const tempoFrames = api.metrics.shot_tempo ?? 0;
  const tempoSeconds = tempoFrames > 0 ? tempoFrames / 30 : 0;
  const drift = api.metrics.torso_drift ?? 0;
  const releaseAngle = api.metrics.release_angle ?? 0;

  const cue = api.mistakes?.[0]?.message ?? (api.made ? "Shot made — reinforce this form." : "Review mechanics on replay.");
  const quality = api.made ? 82 : 48;

  return {
    id: api.shot_id,
    session_id: api.session_id,
    timestamp: new Date(api.timestamp_ms).toISOString(),
    result: api.made ? "made" : "miss",
    coaching_cue: cue,
    quality_score: quality,
    metrics: {
      release_angle: releaseAngle,
      arc: arcDisplay,
      tempo: tempoSeconds,
      drift,
    },
  };
}

function toUiShotDetail(api: ApiShot): ShotDetail {
  const base = toUiShot(api);
  return {
    ...base,
    mistakes: (api.mistakes ?? []).map((m) => m.message),
    context: base.result === "made" ? "Shot made with playable mechanics." : "Miss detected — focus on first fix and replay cues.",
    clip_url: api.clip_url ?? (USE_MOCK ? null : `${BASE_URL}/shots/${api.shot_id}/clip`),
  };
}

function drillToUi(drill?: ApiDrill | null): Drill {
  if (!drill) {
    return {
      name: "No drill available",
      description: "No drill returned for this shot.",
      link: "",
    };
  }
  return {
    name: drill.name,
    description: drill.steps.join(" "),
    link: drill.links?.[0] ?? "",
  };
}

function toUiReplay(api: ApiReplay): ReplayAnalysis {
  return {
    what_went_well: api.what_went_well ?? [],
    what_to_fix: api.what_to_fix ?? [],
    moment_annotations: (api.moment_annotations ?? []).map((m) => ({
      timestamp_s: m.t_sec,
      note: `${m.tag}: ${m.observation} → ${m.correction}`,
    })),
    primary_drill: drillToUi(api.drill),
    backup_drill: drillToUi(api.backup_drill),
    next_shot_focus: api.next_shot_focus ?? "Focus on your first fix cue on the next rep.",
  };
}

let _mockShotCounter = 0;

function rand(min: number, max: number, decimals = 1) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

export function generateMockShot(sessionId: string): Shot {
  _mockShotCounter++;
  const made = Math.random() > 0.42;
  return {
    id: `shot-${Date.now()}-${_mockShotCounter}`,
    session_id: sessionId,
    timestamp: new Date().toISOString(),
    result: made ? "made" : "miss",
    coaching_cue: made ? "Great lift and follow-through." : "Arc was flat; add lift on release.",
    quality_score: made ? rand(68, 97, 0) : rand(28, 65, 0),
    metrics: {
      release_angle: made ? rand(50, 60) : rand(43, 52),
      arc: made ? rand(44, 50) : rand(37, 44),
      tempo: made ? rand(0.75, 0.92) : rand(0.60, 0.80),
      drift: made ? rand(-2, 2) : rand(-8, 8),
    },
  };
}

export function generateMockShotDetail(shot: Shot): ShotDetail {
  return {
    ...shot,
    mistakes: shot.result === "miss" ? ["Arc too flat", "Tempo rushed"] : [],
    context: shot.result === "made" ? "Shot shape stayed consistent." : "Release phase needs correction.",
    clip_url: null,
  };
}

export async function checkHealth(): Promise<HealthResponse> {
  if (USE_MOCK) {
    return { status: "ok", service: "purearc-api-mock" };
  }
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) {
    return { status: "offline" };
  }
  const data = await res.json();
  return { status: data.ok ? "ok" : "offline", service: data.service };
}

export async function startSession(req: SessionStartRequest): Promise<SessionStartResponse> {
  if (USE_MOCK) {
    return {
      session_id: `sess-${Date.now()}`,
      ws_url: `ws://localhost:8000/session/sess-demo/events`,
    };
  }
  const res = await fetch(`${BASE_URL}/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error("Failed to start session");
  return res.json();
}

export async function getLatestSession(): Promise<SessionSummary | null> {
  if (USE_MOCK) {
    return null;
  }

  const res = await fetch(`${BASE_URL}/session/latest`);
  if (!res.ok) throw new Error("Failed to fetch latest session");
  const payload = await res.json();
  return payload?.session ?? null;
}

export async function getSessionShots(sessionId: string): Promise<Shot[]> {
  if (USE_MOCK) {
    return [];
  }
  const res = await fetch(`${BASE_URL}/session/${sessionId}/shots`);
  if (!res.ok) throw new Error("Failed to fetch shots");
  const items = (await res.json()) as Array<{ shot_id: string }>;

  const detailResults = await Promise.all(
    items.map(async (item) => {
      try {
        const detail = await getShotDetail(item.shot_id);
        return detail;
      } catch {
        return null;
      }
    })
  );

  return detailResults.filter((s): s is ShotDetail => Boolean(s));
}

export async function getShotDetail(shotId: string): Promise<ShotDetail> {
  if (USE_MOCK) {
    throw new Error("USE_MOCK_SHOT");
  }
  const res = await fetch(`${BASE_URL}/shots/${shotId}`);
  if (!res.ok) throw new Error("Failed to fetch shot detail");
  const data = (await res.json()) as ApiShot;
  return toUiShotDetail(data);
}

export async function getShotClip(shotId: string): Promise<{ clip_url: string }> {
  if (USE_MOCK) {
    throw new Error("Clip not available in demo mode");
  }
  const res = await fetch(`${BASE_URL}/shots/${shotId}/clip`, { method: "HEAD" });
  if (!res.ok) throw new Error("Clip not available");
  return { clip_url: `${BASE_URL}/shots/${shotId}/clip` };
}

export async function analyzeReplay(
  shotId: string,
  options: { include_drill: boolean; detail_level: "high" | "standard" | "detailed" }
): Promise<ReplayAnalysis> {
  if (USE_MOCK) {
    return {
      what_went_well: ["Consistent shot pocket depth"],
      what_to_fix: ["Flat arc — below ideal launch shape"],
      moment_annotations: [],
      primary_drill: {
        name: "One-Hand Form Shooting",
        description: "Shoot with your shooting hand only and finish high.",
        link: "https://www.youtube.com/results?search_query=one+hand+form+shooting+drill",
      },
      backup_drill: {
        name: "Catch and Shoot",
        description: "Emphasize rhythm and clean release timing.",
        link: "https://www.youtube.com/results?search_query=catch+and+shoot+basketball+drill",
      },
      next_shot_focus: "Lift the arc on your next rep.",
    };
  }

  const detailLevel = options.detail_level === "detailed" ? "high" : options.detail_level;
  const res = await fetch(`${BASE_URL}/shots/${shotId}/replay-analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_drill: options.include_drill, detail_level: detailLevel }),
  });
  if (!res.ok) throw new Error("Replay analysis failed");
  const data = (await res.json()) as ApiReplay;
  return toUiReplay(data);
}

export { USE_MOCK };

/**
 * Send a camera frame (JPEG blob) to the backend for CV processing.
 * Fire-and-forget — errors are silently ignored so the stream keeps going.
 */
export async function sendFrame(
  sessionId: string,
  blob: Blob
): Promise<void> {
  if (USE_MOCK) return;
  const form = new FormData();
  form.append("frame", blob, "frame.jpg");
  try {
    await fetch(`${BASE_URL}/session/${sessionId}/frame`, {
      method: "POST",
      body: form,
    });
  } catch {
    // silently ignore — next frame will retry
  }
}
