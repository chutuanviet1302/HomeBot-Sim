// Data access: the FastAPI backend, or the JSON dump in public/data/ for the static build.
declare const __STATIC__: boolean;
export const STATIC: boolean = __STATIC__;

export interface Card {
  id: string;
  label: string;
  value: string;
  compare?: string | null;
  pct?: number | null;
  sub?: string;
  tone?: "good" | "bad" | "neutral";
}

export interface Summary {
  n: number;
  success: number;
  success_rate: number | null;
  person_contacts: number;
  person_contacts_robot_moving: number;
  runs_with_contact: number;
  min_person_dist_median: number | null;
  min_person_dist_min: number | null;
  arm_stops: number;
  arm_contacts: number;
  furniture_contacts: number;
  falls: number;
  time_s_mean_success: number | null;
  basin_err_mm_mean: number | null;
  perception_err_mm_mean: number | null;
  failed_at?: Record<string, number>;
}

export interface Kpi {
  cards: Card[];
  safety: { with: Summary; without: Summary; min_dist_with: number[]; min_dist_without: number[] } | null;
  policies: (Summary & { experiment: string; policy: string })[];
  has_learned: boolean;
  e5: { groups: (Summary & { label: string })[]; reference_e3: Summary | null } | null;
  levels: (Summary & { experiment: string })[];
}

export interface Experiment extends Summary {
  id: string;
  name: string;
  description: string;
  source: string;
  policies: string[];
}

export interface Run {
  id: number;
  experiment: string;
  seed: number;
  policy: string;
  success: boolean | null;
  failed_at: string | null;
  time_s: number | null;
  person_contacts: number | null;
  person_contacts_robot_moving: number | null;
  min_person_dist: number | null;
  furniture_contacts: number | null;
  arm_contacts: number | null;
  arm_stops: number | null;
  fell: boolean | null;
  basin_err_mm: number | null;
  sink_offset: number | null;
  perception_err_mm: number | null;
  extra: Record<string, unknown>;
}

export interface ReplayInfo {
  key: string;
  level: number;
  seed: number;
  variant: string;
  result: string;
  duration_s: number;
  sink_offset: number;
  person_profile: string;
  policy: string;
}

export interface Shape {
  name: string;
  kind: "box" | "cyl";
  x: number;
  y: number;
  hx?: number;
  hy?: number;
  r?: number;
  yaw?: number;
  h: number;
  movable: boolean;
  wall: boolean;
}

export interface Floorplan {
  width: number;
  height: number;
  items: Shape[];
  places: Record<string, { x: number; y: number; yaw: number }>;
  sink: { x: number; y: number; hx: number; hy: number };
  doorways: number[][];
  robot_radius: number;
  person_radius: number;
  pet_half: number[];
}

type XY = { x: number[]; y: number[] };
export interface Replay extends ReplayInfo {
  hz: number;
  n: number;
  command: string;
  robot: XY & { yaw: number[] };
  can: XY & { z: number[] };
  person: XY | null;
  pet: (XY & { yaw: number[] }) | null;
  state_labels: string[];
  state: number[];
  safety_labels: string[];
  safety: number[];
  floorplan: Floorplan;
}

export interface Job {
  id: number;
  level: number;
  seed: number;
  sink_range: number;
  person: string;
  policy: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  sim_t: number;
  log_tail: string[];
  replay_key: string | null;
  result: string | null;
}

async function getJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const cache = new Map<string, Promise<unknown>>();
function cached<T>(url: string): Promise<T> {
  if (!cache.has(url)) {
    const p = getJSON<T>(url);
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url) as Promise<T>;
}

export const api = {
  kpi: () => cached<Kpi>(STATIC ? "data/kpi.json" : "/api/kpi"),
  experiments: () => cached<Experiment[]>(STATIC ? "data/experiments.json" : "/api/experiments"),
  runs: () => cached<Run[]>(STATIC ? "data/runs.json" : "/api/runs"),
  replays: () => (STATIC ? cached<ReplayInfo[]>("data/replays.json") : getJSON<ReplayInfo[]>("/api/replays")),
  replay: (r: Pick<ReplayInfo, "key" | "level" | "seed" | "variant">) =>
    cached<Replay>(
      STATIC
        ? `data/replay_${r.key}.json`
        : `/api/replays/${r.level}/${r.seed}?variant=${encodeURIComponent(r.variant)}`,
    ),
  media: () => cached<string[]>(STATIC ? "data/media.json" : "/api/media"),
  policies: () => getJSON<string[]>("/api/policies"),
  jobs: () => getJSON<Job[]>("/api/jobs"),
  job: (id: number) => getJSON<Job>(`/api/jobs/${id}`),
  createJob: (body: { level: number; seed: number; sink_range: number; person: string; policy: string }) =>
    getJSON<Job>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export const media = (path: string) => (STATIC ? `media/${path}` : `/media/${path}`);

// ---- formatting / labels
export const fmt = (v: number | null | undefined, digits = 2, unit = "") =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : `${v.toFixed(digits)}${unit}`;
export const pct = (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(0)}%`);

const STATE_VI: Record<string, string> = {
  "NAVIGATE_TO(table)": "Đi tới bàn",
  "MANIPULATE(pick_cup)": "Nhặt lon",
  "NAVIGATE_TO(sink)": "Mang lon tới bồn rửa",
  "MANIPULATE(place_cup)": "Đặt lon vào bồn",
  DONE: "Hoàn thành",
};
export const stateVi = (s: string | null | undefined) => (s ? STATE_VI[s] ?? s : "–");

const SAFETY_VI: Record<string, string> = {
  clear: "thông thoáng",
  slow: "giảm tốc",
  stop: "dừng lại",
  evade: "né tránh",
};
export const safetyVi = (s: string) => SAFETY_VI[s] ?? s;

export const personVi = (p: string) => (p === "elderly" ? "người cao tuổi" : "người trưởng thành");
