"use strict";

// Mirrors imaging._colour on the server: golden-ratio hue stepping, so the
// swatch in the list matches the box drawn on the image. HSV(h, 0.75, 1)
// is exactly hsl(h, 100%, 62.5%).
function classColour(classId) {
  const hue = ((classId * 0.618033988749895) % 1) * 360;
  return `hsl(${hue.toFixed(1)}, 100%, 62.5%)`;
}

function renderDetections(list, detections) {
  list.replaceChildren();
  for (const det of detections) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = classColour(det.class_id);
    name.append(swatch, document.createTextNode(det.label));
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `${(det.score * 100).toFixed(1)}%`;
    item.append(name, score);
    list.append(item);
  }
}

// -- upload -----------------------------------------------------------------

const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const dropzoneLabel = document.getElementById("dropzone-label");
const detectButton = document.getElementById("detect-button");
const uploadError = document.getElementById("upload-error");
const uploadResult = document.getElementById("upload-result");
const uploadImage = document.getElementById("upload-image");
const uploadCaption = document.getElementById("upload-caption");
const uploadDetections = document.getElementById("upload-detections");

let objectUrl = null;

function selectFile(file) {
  if (!file) return;
  fileInput.files = (() => {
    const transfer = new DataTransfer();
    transfer.items.add(file);
    return transfer.files;
  })();
  dropzoneLabel.textContent = file.name;
  detectButton.disabled = false;
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

for (const type of ["dragenter", "dragover"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("hover");
  });
}
for (const type of ["dragleave", "drop"]) {
  dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove("hover");
  });
}
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  detectButton.disabled = true;
  detectButton.textContent = "Detecting…";
  uploadError.hidden = true;

  const body = new FormData();
  body.append("image", file);

  try {
    // Two calls rather than one: the annotated JPEG is what you look at, the
    // JSON is what the list is built from. Both run on the same accelerator,
    // so this costs two inferences -- fine for a hand-driven upload, and it
    // keeps the API honest (an image endpoint that returns image bytes).
    const [annotated, json] = await Promise.all([
      fetch("/api/detect/annotated", { method: "POST", body }),
      fetch("/api/detect", { method: "POST", body }),
    ]);

    if (!annotated.ok || !json.ok) {
      const failed = annotated.ok ? json : annotated;
      const detail = await failed.json().catch(() => ({}));
      throw new Error(detail.detail || `${failed.status} ${failed.statusText}`);
    }

    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(await annotated.blob());
    uploadImage.src = objectUrl;
    uploadResult.hidden = false;

    const result = await json.json();
    uploadCaption.textContent =
      `${result.count} detection${result.count === 1 ? "" : "s"} · ` +
      `${result.inference_ms} ms · ${result.image.width}×${result.image.height}`;
    renderDetections(uploadDetections, result.detections);
  } catch (error) {
    uploadError.textContent = error.message;
    uploadError.hidden = false;
  } finally {
    detectButton.disabled = false;
    detectButton.textContent = "Detect";
  }
});

// -- status polling ---------------------------------------------------------

const devicePill = document.getElementById("device-pill");
const footerModel = document.getElementById("footer-model");
const streamNote = document.getElementById("stream-note");
const streamImage = document.getElementById("stream-image");
const streamDetections = document.getElementById("stream-detections");

let streamStarted = false;

async function poll() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    const engine = status.engine;

    devicePill.className = `pill ${engine.ready ? "pill-ok" : "pill-bad"}`;
    devicePill.textContent = engine.ready
      ? `device ready · ${engine.last_inference_ms ?? "—"} ms · ${engine.frames} frames`
      : `device unavailable: ${engine.error ?? "unknown"}`;

    const size = engine.input_size ? `${engine.input_size.width}×${engine.input_size.height}` : "—";
    footerModel.textContent = `${status.config.hef_path} · ${size} · threshold ${status.config.score_threshold}`;

    const stream = status.stream;
    if (!stream.enabled) {
      streamNote.innerHTML = "No RTSP source configured — set <code>RTSP_URL</code> to enable.";
      return;
    }

    streamNote.textContent = stream.connected
      ? `${stream.url} · ${stream.frames_detected} frames detected · ${stream.frames_dropped} dropped`
      : `${stream.url} · reconnecting${stream.last_error ? ` — ${stream.last_error}` : "…"}`;

    // Only attach the MJPEG source once: resetting src restarts the
    // multipart response and drops the connection mid-frame.
    if (stream.connected && !streamStarted) {
      streamImage.src = "/api/stream/mjpeg";
      streamImage.hidden = false;
      streamStarted = true;
    }
    renderDetections(streamDetections, stream.detections ?? []);
  } catch {
    devicePill.className = "pill pill-bad";
    devicePill.textContent = "service unreachable";
  }
}

poll();
setInterval(poll, 2000);
