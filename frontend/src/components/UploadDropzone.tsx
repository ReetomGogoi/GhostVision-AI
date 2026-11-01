import { useRef, useState } from "react";

import { ImageIcon, UploadIcon } from "./Icons";

interface Props {
  file: File | null;
  previewUrl: string | null;
  disabled?: boolean;
  onFile: (file: File) => void;
  onError: (message: string) => void;
}

const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const MAX_BYTES = 15 * 1024 * 1024;

export default function UploadDropzone({
  file,
  previewUrl,
  disabled,
  onFile,
  onError,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function choose(files: FileList | null) {
    const candidate = files?.[0];
    if (!candidate) return;
    if (!ALLOWED_TYPES.has(candidate.type)) {
      onError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      onError("The image is larger than the 15 MB limit.");
      return;
    }
    onFile(candidate);
  }

  return (
    <div className="upload-group">
      <div
        className={`dropzone ${dragging ? "is-dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(event) => {
          if (!disabled && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (!disabled) choose(event.dataTransfer.files);
        }}
      >
        {previewUrl ? (
          <>
            <img className="dropzone-preview" src={previewUrl} alt="Selected upload preview" />
            <div className="dropzone-preview-shade" />
            <div className="dropzone-copy dropzone-copy-over">
              <ImageIcon className="dropzone-icon" />
              <strong>{file?.name}</strong>
              <span>Click or drop to replace</span>
            </div>
          </>
        ) : (
          <div className="dropzone-copy">
            <span className="upload-glyph"><UploadIcon /></span>
            <strong>Drop a wildlife image here</strong>
            <span>or browse from your device</span>
            <small>JPEG, PNG or WebP · up to 15 MB</small>
          </div>
        )}
      </div>
      <button
        type="button"
        className="camera-button"
        onClick={() => cameraRef.current?.click()}
        disabled={disabled}
      >
        Use camera
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(event) => choose(event.target.files)}
      />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        hidden
        onChange={(event) => choose(event.target.files)}
      />
    </div>
  );
}
