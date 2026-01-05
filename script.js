const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const output = document.getElementById("output");
const buttons = document.querySelectorAll("button[data-model]");
const ctx = preview.getContext("2d");
let imageDataUrl = null;

function setStatus(message, tone) {
  const color = tone === "error" ? "#ffb3b3" : "#f3f3f3";
  output.innerHTML = `
    <p class="status" style="color:${color}">${message}</p>
  `;
}

if (window.location.protocol === "file:") {
  setStatus("Open this page via `python app.py` so the /predict endpoints are available.", "error");
}

function drawPlaceholder() {
  ctx.fillStyle = "#f0ece6";
  ctx.fillRect(0, 0, preview.width, preview.height);
  ctx.fillStyle = "#8a7f72";
  ctx.font = "16px IBM Plex Sans";
  ctx.textAlign = "center";
  ctx.fillText("Preview", preview.width / 2, preview.height / 2);
}

drawPlaceholder();

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) {
    imageDataUrl = null;
    drawPlaceholder();
    setStatus("Awaiting input. Upload an image to begin.");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    imageDataUrl = reader.result;
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, preview.width, preview.height);
      const scale = Math.min(preview.width / img.width, preview.height / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      const x = (preview.width - w) / 2;
      const y = (preview.height - h) / 2;
      ctx.drawImage(img, x, y, w, h);
    };
    img.src = imageDataUrl;
    setStatus("Image loaded. Choose a model to run.");
  };
  reader.readAsDataURL(file);
});

async function runInference(model) {
  if (!imageDataUrl) {
    setStatus("Upload an image first.", "error");
    return;
  }

  setStatus(`Sending image to ${model.toUpperCase()}...`);
  try {
    const response = await fetch(`/predict/${model}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl })
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = data.error
        ? `Server error: ${data.error}. Train models with: python app.py --train`
        : "Prediction service returned an error. Ensure the server is running.";
      setStatus(message, "error");
      return;
    }

    const label = data.label || "unknown";
    const confidence = data.confidence !== undefined
      ? `${Math.round(data.confidence * 100)}%`
      : "n/a";
    const resizeNote = data.resized ? "<p>Image resized to 32x32 for inference.</p>" : "";

    output.innerHTML = `
      <p class="status">Prediction: <strong>${label}</strong></p>
      <p>Confidence: ${confidence}</p>
      ${resizeNote}
    `;
  } catch (error) {
    setStatus(
      "No prediction service found. Run `python app.py` and retry.",
      "error"
    );
  }
}

buttons.forEach((button) => {
  button.addEventListener("click", () => {
    const model = button.dataset.model;
    runInference(model);
  });
});
