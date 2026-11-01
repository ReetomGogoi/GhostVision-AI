import { afterEach, describe, expect, it, vi } from "vitest";

import {
  getHealth,
  pngDataUrl,
  predict,
  type HealthResponse,
  type RgbPrediction,
  type ThermalPrediction,
} from "./api";

function jsonResponse(payload: object, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("GhostVision API client", () => {
  it("reads the committed health schema from /health", async () => {
    const payload: HealthResponse = {
      status: "ok",
      modalities: {
        rgb: {
          configured: true,
          models: {
            baseline: { configured: true, checkpoint_available: true },
            augmented: { configured: true, checkpoint_available: true },
          },
        },
        thermal: { configured: true, checkpoint_available: true },
      },
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getHealth()).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("/health", { signal: undefined });
  });

  it("posts RGB uploads to /predict/rgb with model_name", async () => {
    const payload: RgbPrediction = {
      modality: "rgb",
      model_name: "baseline",
      mask: { media_type: "image/png", encoding: "base64", width: 2, height: 2, data: "bWFzaw==" },
      overlay: { media_type: "image/png", encoding: "base64", width: 2, height: 2, data: "b3ZlcmxheQ==" },
      metadata: { checkpoint: "baseline.pth", threshold: 0.5 },
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["rgb"], "scene.png", { type: "image/png" });

    await expect(predict(file, "rgb", "baseline")).resolves.toEqual(payload);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    const form = options.body as FormData;
    expect(url).toBe("/predict/rgb");
    expect(options.method).toBe("POST");
    expect(form.get("image")).toBe(file);
    expect(form.get("model_name")).toBe("baseline");
    expect(form.has("modality")).toBe(false);
    expect(form.has("rgb_model")).toBe(false);
  });

  it("posts thermal uploads to /predict/thermal without RGB fields", async () => {
    const payload: ThermalPrediction = {
      modality: "thermal",
      animal_count: 1,
      detections: [{ class_name: "animal", confidence: 0.81, bbox: [1, 2, 8, 9] }],
      metadata: { checkpoint: "best.pt", confidence_threshold: 0.25 },
    };
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["thermal"], "thermal.png", { type: "image/png" });

    await expect(predict(file, "thermal", "augmented")).resolves.toEqual(payload);
    const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    const form = options.body as FormData;
    expect(url).toBe("/predict/thermal");
    expect(form.get("image")).toBe(file);
    expect(form.has("model_name")).toBe(false);
  });

  it("turns backend PNG payloads into displayable data URLs", () => {
    expect(pngDataUrl({ media_type: "image/png", encoding: "base64", width: 1, height: 1, data: "YWJj" }))
      .toBe("data:image/png;base64,YWJj");
  });

  it("surfaces the backend's safe error detail", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ detail: "Checkpoint unavailable." }, 503)));
    const file = new File(["rgb"], "scene.png", { type: "image/png" });
    await expect(predict(file, "rgb", "augmented")).rejects.toThrow("Checkpoint unavailable.");
  });
});
