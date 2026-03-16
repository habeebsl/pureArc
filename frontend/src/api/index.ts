const BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL ?? "";
const USE_MOCK = !BASE_URL;

export { USE_MOCK };

// ── Types ─────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok" | "offline";
  service?: string;
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
  duration_min: number;
  steps: string[];
  links: string[];
}

export interface ReplayAnalysis {
  what_went_well: string[];
  what_to_fix: string[];
  moment_annotations: Array<{ timestamp_s: number; note: string }>;
  primary_drill: Drill;
  backup_drill: Drill;
  next_shot_focus: string;
}

export interface VideoAnalysisResult {
  session_id: string;
  total_shots: number;
  makes: number;
  shots: Shot[];
  drills: Drill[];
  summary: string;
}

// ── Internal API shape ────────────────────────────────────────────────────────

interface ApiShotMetrics {
  release_angle: number | null;
  arc_height_ratio: number | null;
  shot_tempo: number | null;
  torso_drift: number | null;
}

interface ApiMistake {
  tag: string;
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
  tag: string;
  observation: string;
  correction: string;
}

interface ApiReplay {
  what_went_well: string[];
  what_to_fix: string[];
  moment_annotations?: ApiMoment[];
  drill?: ApiDrill | null;
  backup_drill?: ApiDrill | null;
  next_shot_focus?: string | null;
}

interface ApiVideoAnalysis {
  session_id: string;
  total_shots: number;
  makes: number;
  shots: ApiShot[];
  drills: ApiDrill[];
  summary: string;
}

// ── Converters ────────────────────────────────────────────────────────────────

function toUiShot(api: ApiShot): Shot {
  const arcRatio     = api.metrics.arc_height_ratio ?? 0;
  const arcDisplay   = Math.max(0, Math.min(100, arcRatio * 100));
  const tempoFrames  = api.metrics.shot_tempo ?? 0;
  const tempoSeconds = tempoFrames > 0 ? tempoFrames / 30 : 0;
  const drift        = api.metrics.torso_drift ?? 0;
  const releaseAngle = api.metrics.release_angle ?? 0;
  const cue          = api.mistakes?.[0]?.message ?? (api.made ? "Shot made — reinforce this form." : "Review mechanics on replay.");
  const quality      = api.made ? 82 : 48;

  return {
    id:           api.shot_id,
    session_id:   api.session_id,
    timestamp:    new Date(api.timestamp_ms).toISOString(),
    result:       api.made ? "made" : "miss",
    coaching_cue: cue,
    quality_score: quality,
    metrics: {
      release_angle: releaseAngle,
      arc:           arcDisplay,
      tempo:         tempoSeconds,
      drift,
    },
  };
}

function toUiShotDetail(api: ApiShot): ShotDetail {
  const base = toUiShot(api);
  return {
    ...base,
    mistakes:  (api.mistakes ?? []).map((m) => m.message),
    context:   base.result === "made" ? "Shot made with playable mechanics." : "Miss detected — focus on first fix and replay cues.",
    clip_url:  api.clip_url ?? null,
  };
}

function toUiDrill(d: ApiDrill): Drill {
  return { name: d.name, duration_min: d.duration_min, steps: d.steps, links: d.links ?? [] };
}

function toUiReplay(api: ApiReplay): ReplayAnalysis {
  return {
    what_went_well:       api.what_went_well ?? [],
    what_to_fix:          api.what_to_fix ?? [],
    moment_annotations:   (api.moment_annotations ?? []).map((m) => ({
      timestamp_s: m.t_sec,
      note: `${m.tag}: ${m.observation} → ${m.correction}`,
    })),
    primary_drill: toUiDrill(api.drill ?? { name: "Form Shooting", duration_min: 8, steps: [], links: [] }),
    backup_drill:  toUiDrill(api.backup_drill ?? { name: "Partner Shooting", duration_min: 8, steps: [], links: [] }),
    next_shot_focus: api.next_shot_focus ?? "Focus on your first fix cue on the next rep.",
  };
}

// ── Mock data ─────────────────────────────────────────────────────────────────

let _mockShotCounter = 0;

function rand(min: number, max: number, decimals = 1) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

export function generateMockShotDetail(shot: Shot): ShotDetail {
  return {
    ...shot,
    mistakes: shot.result === "miss" ? ["Arc too flat", "Tempo rushed"] : [],
    context:  shot.result === "made" ? "Shot shape stayed consistent." : "Release phase needs correction.",
    clip_url: null,
  };
}

function _generateMockAnalysis(): VideoAnalysisResult {
  const shots: Shot[] = Array.from({ length: 5 }, (_, i) => {
    _mockShotCounter++;
    const made = Math.random() > 0.45;
    return {
      id:           `shot-mock-${i + 1}`,
      session_id:   "mock-session",
      timestamp:    new Date().toISOString(),
      result:       made ? "made" : "miss",
      coaching_cue: made ? "Great lift and follow-through." : "Arc was flat; add lift on release.",
      quality_score: made ? rand(68, 97, 0) : rand(28, 65, 0),
      metrics: {
        release_angle: made ? rand(50, 60) : rand(43, 52),
        arc:           made ? rand(44, 50) : rand(37, 44),
        tempo:         made ? rand(0.75, 0.92) : rand(0.60, 0.80),
        drift:         made ? rand(-2, 2) : rand(-8, 8),
      },
    } as Shot;
  });

  const makes = shots.filter((s) => s.result === "made").length;

  return {
    session_id:   "mock-session",
    total_shots:  shots.length,
    makes,
    shots,
    drills: [
      { name: "Off-the-Dribble Form Shooting", duration_min: 10, steps: ["Use controlled 1-2 footwork off dribble.", "Focus on balanced rise and high finish.", "Do 3 sets of 12 pull-up reps."], links: [] },
      { name: "Balance and Hold Drill",         duration_min: 8,  steps: ["Hold your release follow-through for 3 seconds.", "Check alignment on finish.", "Repeat for 3 sets of 10 makes."], links: [] },
    ],
    summary: `${makes} of ${shots.length} shots made in this clip.`,
  };
}

// ── Public API functions ──────────────────────────────────────────────────────

export async function checkHealth(): Promise<HealthResponse> {
  if (USE_MOCK) return { status: "ok", service: "purearc-api-mock" };
  try {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) return { status: "offline" };
    const data = await res.json();
    return { status: data.ok ? "ok" : "offline", service: data.service };
  } catch {
    return { status: "offline" };
  }
}

export async function uploadVideo(file: File): Promise<VideoAnalysisResult> {
  if (USE_MOCK) {
    // Simulate processing delay in demo mode
    await new Promise((resolve) => setTimeout(resolve, 2000));
    return _generateMockAnalysis();
  }

  const form = new FormData();
  form.append("video", file, file.name);

  const res = await fetch(`${BASE_URL}/analyze-video`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Analysis failed (${res.status}): ${text.slice(0, 200)}`);
  }

  const data = (await res.json()) as ApiVideoAnalysis;
  return {
    session_id:  data.session_id,
    total_shots: data.total_shots,
    makes:       data.makes,
    shots:       data.shots.map(toUiShot),
    drills:      (data.drills ?? []).map(toUiDrill),
    summary:     data.summary,
  };
}

export async function getShotDetail(shotId: string): Promise<ShotDetail> {
  if (USE_MOCK) throw new Error("USE_MOCK_SHOT");
  const res = await fetch(`${BASE_URL}/shots/${shotId}`);
  if (!res.ok) throw new Error("Failed to fetch shot detail");
  return toUiShotDetail((await res.json()) as ApiShot);
}

export async function analyzeReplay(
  shotId: string,
  options: { include_drill: boolean; detail_level: "high" | "standard" | "detailed" },
): Promise<ReplayAnalysis> {
  if (USE_MOCK) {
    return {
      what_went_well:     ["Consistent shot pocket depth"],
      what_to_fix:        ["Flat arc — below ideal launch shape"],
      moment_annotations: [],
      primary_drill: {
        name: "One-Hand Form Shooting",
        duration_min: 10,
        steps: ["Shoot with dominant hand only.", "Focus on finger-tip release and arc."],
        links: ["https://www.youtube.com/results?search_query=one+hand+form+shooting+drill"],
      },
      backup_drill: {
        name: "Catch and Shoot",
        duration_min: 8,
        steps: ["Emphasize rhythm and clean release timing."],
        links: ["https://www.youtube.com/results?search_query=catch+and+shoot+basketball+drill"],
      },
      next_shot_focus: "Lift the arc on your next rep.",
    };
  }

  const detailLevel = options.detail_level === "detailed" ? "high" : options.detail_level;
  const res = await fetch(`${BASE_URL}/shots/${shotId}/replay-analysis`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ include_drill: options.include_drill, detail_level: detailLevel }),
  });
  if (!res.ok) throw new Error("Replay analysis failed");
  return toUiReplay((await res.json()) as ApiReplay);
}
