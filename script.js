const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const output = document.getElementById("output");
const buttons = document.querySelectorAll("button[data-model]");
const saveButton = document.getElementById("saveImage");
const labelInput = document.getElementById("imageLabel");
const galleryGrid = document.getElementById("galleryGrid");
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
loadGallery();

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

async function saveToGallery() {
  if (!imageDataUrl) {
    setStatus("Upload an image first.", "error");
    return;
  }
  setStatus("Saving image to gallery...");
  try {
    const response = await fetch("/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageDataUrl, label: labelInput.value })
    });
    const data = await response.json();
    if (!response.ok) {
      const message = data.error ? `Server error: ${data.error}` : "Failed to save image.";
      setStatus(message, "error");
      return;
    }
    setStatus("Saved. Choose a model to run.");
    labelInput.value = "";
    loadGallery();
  } catch (error) {
    setStatus("Unable to save image. Ensure the server is running.", "error");
  }
}

async function loadGallery() {
  try {
    const response = await fetch("/uploads");
    const data = await response.json();
    if (!response.ok) {
      return;
    }
    renderGallery(data.items || []);
  } catch (error) {
    // Ignore if server is offline.
  }
}

function renderGallery(items) {
  galleryGrid.innerHTML = "";
  if (!items.length) {
    galleryGrid.innerHTML = "<p class=\"status\">No saved images yet.</p>";
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "card gallery-card";
    const img = document.createElement("img");
    img.src = `/uploaded_images/${item.filename}`;
    img.alt = item.label || "uploaded image";
    const meta = document.createElement("div");
    meta.className = "gallery-meta";
    const label = item.label ? `Label: ${item.label}` : "Label: none";
    meta.textContent = `${label} · ${new Date(item.created_at).toLocaleString()}`;
    const useButton = document.createElement("button");
    useButton.className = "btn ghost";
    useButton.textContent = "Use Image";
    useButton.addEventListener("click", async () => {
      imageDataUrl = null;
      const response = await fetch(img.src);
      const blob = await response.blob();
      const reader = new FileReader();
      reader.onload = () => {
        imageDataUrl = reader.result;
        const previewImg = new Image();
        previewImg.onload = () => {
          ctx.clearRect(0, 0, preview.width, preview.height);
          const scale = Math.min(preview.width / previewImg.width, preview.height / previewImg.height);
          const w = previewImg.width * scale;
          const h = previewImg.height * scale;
          const x = (preview.width - w) / 2;
          const y = (preview.height - h) / 2;
          ctx.drawImage(previewImg, x, y, w, h);
        };
        previewImg.src = imageDataUrl;
        setStatus("Loaded saved image. Choose a model to run.");
      };
      reader.readAsDataURL(blob);
    });
    card.appendChild(img);
    card.appendChild(meta);
    const deleteButton = document.createElement("button");
    deleteButton.className = "btn danger";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async () => {
      if (!window.confirm("Delete this saved image?")) {
        return;
      }
      try {
        const response = await fetch(`/uploads/${item.filename}`, { method: "DELETE" });
        if (!response.ok) {
          setStatus("Delete failed. Try again.", "error");
          return;
        }
        loadGallery();
      } catch (error) {
        setStatus("Delete failed. Ensure the server is running.", "error");
      }
    });
    card.appendChild(useButton);
    card.appendChild(deleteButton);
    galleryGrid.appendChild(card);
  });
}

saveButton.addEventListener("click", saveToGallery);
