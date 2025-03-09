export type ChangeClass =
  | "no_change"
  | "vegetation_loss"
  | "urban_expansion"
  | "water_change"
  | "agricultural_change";

export interface Scene {
  id: number;
  name: string;
  sensor: string;
  region: string | null;
  acquired_before: string | null;
  acquired_after: string | null;
  tile_count: number;
}

export interface Prediction {
  id: number;
  tile_id: number;
  label: ChangeClass;
  confidence: number;
  entropy: number;
  probabilities: Record<ChangeClass, number>;
  model_name: string;
  created_at: string;
}

export interface Annotation {
  id: number;
  tile_id: number;
  prediction_id: number | null;
  analyst: string;
  label: ChangeClass;
  action: "confirmed" | "corrected" | "skipped";
  notes: string | null;
  created_at: string;
}

export interface Tile {
  id: number;
  scene_id: number;
  tile_key: string;
  lat: number | null;
  lon: number | null;
  ground_truth: ChangeClass | null;
  latest_prediction: Prediction | null;
  latest_annotation: Annotation | null;
}

export interface TileDetail extends Tile {
  predictions: Prediction[];
  annotations: Annotation[];
}

export interface Stats {
  tiles: number;
  predicted: number;
  reviewed: number;
  corrected: number;
  agreement_rate: number | null;
  predicted_label_counts: Record<string, number>;
  reviewed_label_counts: Record<string, number>;
  confusion: Record<string, Record<string, number>>;
}

export type QueueStrategy = "uncertainty" | "entropy" | "changes";
