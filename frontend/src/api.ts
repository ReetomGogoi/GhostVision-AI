export type Modality = "rgb" | "thermal";
export type RgbModel = "baseline" | "augmented";

export interface EncodedPng {
  media_type: "image/png";
  encoding: "base64";
  width: number;
  height: number;
  data: string;
}

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
}

interface PredictionBase {
  modality: Modality;
  metadata: Record<string, string | number | boolean | null>;
}

export interface RgbPrediction extends PredictionBase {
  modality: "rgb";
  model_name: RgbModel;
  mask: EncodedPng;
  overlay: EncodedPng;
}

export interface ThermalPrediction extends PredictionBase {
  modality: "thermal";
  animal_count: number;
  detections: Detection[];
}

export type Prediction = RgbPrediction | ThermalPrediction;

export interface CheckpointStatus {
  configured: boolean;
  checkpoint_available: boolean;
}

export interface HealthResponse {
  status: "ok" | string;
  modalities: {
    rgb: {
      configured: boolean;
      models: Record<RgbModel, CheckpointStatus>;
    };
    thermal: CheckpointStatus;
  };
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export function pngDataUrl(image: EncodedPng): string {
  if (image.media_type !== "image/png" || image.encoding !== "base64") {
    throw new Error("The backend returned an unsupported image encoding.");
  }
  return `data:${image.media_type};base64,${image.data}`;
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
  } catch {
    // Use the safe status message below for unstructured server responses.
  }
  return `Request failed with status ${response.status}.`;
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`, { signal });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<HealthResponse>;
}

export async function predict(
  file: File,
  modality: Modality,
  rgbModel: RgbModel,
  signal?: AbortSignal,
): Promise<Prediction> {
  const form = new FormData();
  form.append("image", file);

  const endpoint = modality === "rgb" ? "/predict/rgb" : "/predict/thermal";
  if (modality === "rgb") form.append("model_name", rgbModel);

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<Prediction>;
}
