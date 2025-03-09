import type { Annotation, ChangeClass, Prediction, QueueStrategy, Scene, Stats, Tile, TileDetail } from "./types";

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ? JSON.stringify(body.detail) : `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const qs = (params: Record<string, string | number | undefined | null>) => {
  const p = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v !== undefined && v !== null && p.set(k, String(v)));
  const s = p.toString();
  return s ? `?${s}` : "";
};

export const api = {
  scenes: () => request<Scene[]>("/scenes"),
  classes: () => request<ChangeClass[]>("/classes"),
  models: () => request<{ available: string[] }>("/models"),
  queue: (sceneId: number | null, strategy: QueueStrategy, limit = 50) =>
    request<Tile[]>(`/review/queue${qs({ scene_id: sceneId, strategy, limit })}`),
  tile: (id: number) => request<TileDetail>(`/tiles/${id}`),
  predict: (id: number, model = "auto") =>
    request<Prediction>(`/tiles/${id}/predict`, { method: "POST", body: JSON.stringify({ model }) }),
  predictScene: (sceneId: number, model = "auto") =>
    request<{ predicted: number; model: string }>(`/scenes/${sceneId}/predict`, {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  annotate: (id: number, body: { analyst: string; label: ChangeClass; notes?: string; prediction_id?: number; action?: string }) =>
    request<Annotation>(`/tiles/${id}/annotations`, { method: "POST", body: JSON.stringify(body) }),
  stats: (sceneId: number | null) => request<Stats>(`/stats${qs({ scene_id: sceneId })}`),
  previewUrl: (id: number, which: string) => `${API_URL}/tiles/${id}/preview/${which}.png`,
  exportUrl: () => `${API_URL}/annotations/export`,
};
