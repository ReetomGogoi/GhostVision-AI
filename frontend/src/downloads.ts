import type { Detection } from "./api";

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not load the result image."));
    image.src = url;
  });
}

function downloadCanvas(canvas: HTMLCanvasElement, filename: string): Promise<void> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Could not create the downloadable visualization."));
        return;
      }
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
      resolve();
    }, "image/png");
  });
}

export async function downloadRgbComparison(
  originalUrl: string,
  maskUrl: string,
  overlayUrl: string,
  filename: string,
): Promise<void> {
  const [original, mask, overlay] = await Promise.all([
    loadImage(originalUrl),
    loadImage(maskUrl),
    loadImage(overlayUrl),
  ]);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const gutter = Math.max(8, Math.round(width / 80));
  const canvas = document.createElement("canvas");
  canvas.width = width * 3 + gutter * 2;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  context.fillStyle = "#0a0f12";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(original, 0, 0, width, height);
  context.drawImage(mask, width + gutter, 0, width, height);
  context.drawImage(overlay, width * 2 + gutter * 2, 0, width, height);
  await downloadCanvas(canvas, filename);
}

export async function downloadThermalAnnotation(
  originalUrl: string,
  detections: Detection[],
  filename: string,
): Promise<void> {
  const original = await loadImage(originalUrl);
  const canvas = document.createElement("canvas");
  canvas.width = original.naturalWidth;
  canvas.height = original.naturalHeight;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");
  context.drawImage(original, 0, 0);

  const stroke = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 240));
  const fontSize = Math.max(12, Math.round(Math.min(canvas.width, canvas.height) / 38));
  context.font = `600 ${fontSize}px ui-monospace, monospace`;
  context.lineWidth = stroke;

  detections.forEach((detection, index) => {
    const [rawX1, rawY1, rawX2, rawY2] = detection.bbox;
    const x1 = Math.max(0, Math.min(canvas.width, Math.min(rawX1, rawX2)));
    const y1 = Math.max(0, Math.min(canvas.height, Math.min(rawY1, rawY2)));
    const x2 = Math.max(0, Math.min(canvas.width, Math.max(rawX1, rawX2)));
    const y2 = Math.max(0, Math.min(canvas.height, Math.max(rawY1, rawY2)));
    const label = `${String(index + 1).padStart(2, "0")}  ${(detection.confidence * 100).toFixed(1)}%`;
    const labelWidth = context.measureText(label).width + 12;
    const labelHeight = fontSize + 10;
    const labelY = Math.max(0, y1 - labelHeight);

    context.strokeStyle = "#eebb4a";
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.fillStyle = "#eebb4a";
    context.fillRect(x1, labelY, labelWidth, labelHeight);
    context.fillStyle = "#0b1013";
    context.fillText(label, x1 + 6, labelY + fontSize + 2);
  });

  await downloadCanvas(canvas, filename);
}
