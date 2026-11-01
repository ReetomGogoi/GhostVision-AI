import { useMemo, useState } from "react";

import {
  pngDataUrl,
  type Prediction,
  type RgbPrediction,
  type ThermalPrediction,
} from "../api";
import {
  downloadRgbComparison,
  downloadThermalAnnotation,
} from "../downloads";
import { DownloadIcon } from "./Icons";

type RgbView = "original" | "mask" | "overlay" | "compare";
type ThermalView = "annotated" | "original";

interface Props {
  prediction: Prediction;
  originalUrl: string;
  originalName: string;
  onError: (message: string) => void;
}

function DownloadButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <button
      className="download-button"
      type="button"
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        try {
          await onClick();
        } finally {
          setBusy(false);
        }
      }}
    >
      <DownloadIcon />
      {busy ? "Preparing download" : label}
    </button>
  );
}

function Metadata({ values }: { values: Prediction["metadata"] }) {
  return (
    <dl className="metadata-grid">
      {Object.entries(values).map(([key, value]) => (
        <div key={key}>
          <dt>{key.replaceAll("_", " ")}</dt>
          <dd>{String(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function safeBaseName(filename: string): string {
  return (
    filename
      .replace(/\.[^.]+$/, "")
      .replace(/[^a-zA-Z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "wildlife"
  );
}

function RgbWorkspace({
  prediction,
  originalUrl,
  originalName,
  onError,
}: {
  prediction: RgbPrediction;
  originalUrl: string;
  originalName: string;
  onError: (message: string) => void;
}) {
  const [view, setView] = useState<RgbView>("overlay");
  const [split, setSplit] = useState(52);
  const maskUrl = useMemo(() => pngDataUrl(prediction.mask), [prediction.mask]);
  const overlayUrl = useMemo(
    () => pngDataUrl(prediction.overlay),
    [prediction.overlay],
  );
  const tabs: Array<[RgbView, string]> = [
    ["original", "Original"],
    ["mask", "Mask"],
    ["overlay", "Overlay"],
    ["compare", "Compare"],
  ];
  const images = { original: originalUrl, mask: maskUrl, overlay: overlayUrl };

  async function downloadComparison() {
    try {
      await downloadRgbComparison(
        originalUrl,
        maskUrl,
        overlayUrl,
        `${safeBaseName(originalName)}-rgb-comparison.png`,
      );
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Could not download the RGB comparison.",
      );
    }
  }

  return (
    <>
      <div className="workspace-toolbar">
        <div className="view-tabs" role="tablist" aria-label="RGB result view">
          {tabs.map(([value, label]) => (
            <button
              type="button"
              role="tab"
              aria-selected={view === value}
              className={view === value ? "active" : ""}
              onClick={() => setView(value)}
              key={value}
            >
              {label}
            </button>
          ))}
        </div>
        <DownloadButton
          onClick={downloadComparison}
          label="Download comparison"
        />
      </div>
      <div className="image-stage result-stage">
        {view === "compare" ? (
          <div className="compare-frame">
            <img src={originalUrl} alt="Original uploaded wildlife scene" />
            <div
              className="compare-overlay"
              style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
            >
              <img src={overlayUrl} alt="Segmentation overlay" />
            </div>
            <span className="compare-label compare-label-left">Original</span>
            <span className="compare-label compare-label-right">Overlay</span>
            <div className="compare-line" style={{ left: `${split}%` }}>
              <span />
            </div>
            <input
              className="compare-range"
              type="range"
              min="0"
              max="100"
              value={split}
              onChange={(event) => setSplit(Number(event.target.value))}
              aria-label="Move image comparison divider"
            />
          </div>
        ) : (
          <img
            className={view === "mask" ? "pixelated" : ""}
            src={images[view]}
            alt={`${view} segmentation result`}
          />
        )}
      </div>
      <div className="result-details">
        <div>
          <span className="eyebrow">Inference output</span>
          <h2>Camouflage segmentation complete</h2>
          <p>
            The mask and highlight overlay are the selected checkpoint's
            prediction. The original image remains local to this browser view.
          </p>
        </div>
        <Metadata values={prediction.metadata} />
      </div>
    </>
  );
}

function ThermalImage({
  originalUrl,
  prediction,
  showBoxes,
}: {
  originalUrl: string;
  prediction: ThermalPrediction;
  showBoxes: boolean;
}) {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  return (
    <div className="thermal-image-frame">
      <img
        src={originalUrl}
        alt={
          showBoxes
            ? "Thermal wildlife image with detected animal bounding boxes"
            : "Original thermal wildlife image"
        }
        onLoad={(event) =>
          setDimensions({
            width: event.currentTarget.naturalWidth,
            height: event.currentTarget.naturalHeight,
          })
        }
      />
      {showBoxes && dimensions.width > 0 && dimensions.height > 0 && (
        <svg
          className="thermal-box-overlay"
          viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {prediction.detections.map((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const left = Math.min(x1, x2);
            const top = Math.min(y1, y2);
            const width = Math.abs(x2 - x1);
            const height = Math.abs(y2 - y1);
            const fontSize = Math.max(
              12,
              Math.min(dimensions.width, dimensions.height) / 34,
            );
            return (
              <g key={`${detection.bbox.join("-")}-${index}`}>
                <rect
                  x={left}
                  y={top}
                  width={width}
                  height={height}
                  className="thermal-box"
                  vectorEffect="non-scaling-stroke"
                />
                <text
                  x={left + 5}
                  y={Math.max(fontSize, top - 5)}
                  fontSize={fontSize}
                  className="thermal-box-number"
                >
                  {String(index + 1).padStart(2, "0")}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function ThermalWorkspace({
  prediction,
  originalUrl,
  originalName,
  onError,
}: {
  prediction: ThermalPrediction;
  originalUrl: string;
  originalName: string;
  onError: (message: string) => void;
}) {
  const [view, setView] = useState<ThermalView>("annotated");

  async function downloadAnnotation() {
    try {
      await downloadThermalAnnotation(
        originalUrl,
        prediction.detections,
        `${safeBaseName(originalName)}-thermal-detections.png`,
      );
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Could not download the thermal visualization.",
      );
    }
  }

  return (
    <>
      <div className="workspace-toolbar">
        <div
          className="view-tabs"
          role="tablist"
          aria-label="Thermal result view"
        >
          {(["annotated", "original"] as ThermalView[]).map((value) => (
            <button
              type="button"
              role="tab"
              aria-selected={view === value}
              className={view === value ? "active" : ""}
              onClick={() => setView(value)}
              key={value}
            >
              {value === "annotated" ? "Detections" : "Original"}
            </button>
          ))}
        </div>
        <DownloadButton
          onClick={downloadAnnotation}
          label="Download annotated image"
        />
      </div>
      <div className="image-stage result-stage thermal-stage">
        <ThermalImage
          originalUrl={originalUrl}
          prediction={prediction}
          showBoxes={view === "annotated"}
        />
      </div>
      <div className="thermal-summary">
        <div className="count-card">
          <span>Animals detected</span>
          <strong>{prediction.animal_count}</strong>
        </div>
        <div className="detection-list" aria-label="Thermal detections">
          {prediction.detections.length === 0 ? (
            <p className="empty-detections">
              No detections above the configured confidence threshold.
            </p>
          ) : (
            prediction.detections.map((detection, index) => (
              <div
                className="detection-row"
                key={`${detection.bbox.join("-")}-${index}`}
              >
                <span className="detection-index">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <strong>{detection.class_name}</strong>
                  <small>
                    XYXY · {detection.bbox.map((value) => value.toFixed(1)).join(", ")}
                  </small>
                </div>
                <span className="confidence">
                  {(detection.confidence * 100).toFixed(1)}%
                </span>
              </div>
            ))
          )}
        </div>
        <Metadata values={prediction.metadata} />
      </div>
    </>
  );
}

export default function ResultWorkspace({
  prediction,
  originalUrl,
  originalName,
  onError,
}: Props) {
  return prediction.modality === "rgb" ? (
    <RgbWorkspace
      prediction={prediction}
      originalUrl={originalUrl}
      originalName={originalName}
      onError={onError}
    />
  ) : (
    <ThermalWorkspace
      prediction={prediction}
      originalUrl={originalUrl}
      originalName={originalName}
      onError={onError}
    />
  );
}
