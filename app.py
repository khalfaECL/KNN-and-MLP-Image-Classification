from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

from read_cifar import read_cifar, split_dataset

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
UPLOAD_DIR = APP_DIR / "uploaded_images"
UPLOAD_INDEX = UPLOAD_DIR / "index.json"

CIFAR_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CIFAR_LABELS)}

app = Flask(__name__, static_folder=str(APP_DIR))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
if not UPLOAD_INDEX.exists():
    UPLOAD_INDEX.write_text("[]", encoding="utf-8")


def load_upload_index() -> list[dict]:
    try:
        return json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_upload_index(items: list[dict]) -> None:
    UPLOAD_INDEX.write_text(json.dumps(items, indent=2), encoding="utf-8")


def decode_image_raw(data_url: str) -> tuple[Image.Image, bool, str]:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url
    raw = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        resized = image.size != (32, 32)
        image_format = image.format or "PNG"
        return image.copy(), resized, image_format


def decode_image(data_url: str) -> tuple[np.ndarray, bool]:
    image, resized, _ = decode_image_raw(data_url)
    if resized:
        image = ImageOps.fit(image, (32, 32), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return array.reshape(1, -1), resized


def softmax(x: np.ndarray) -> np.ndarray:
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)


def load_knn_model():
    path = MODEL_DIR / "knn.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "data_train": data["data_train"],
        "labels_train": data["labels_train"],
        "k": int(data["k"]),
        "mean": data["mean"] if "mean" in data else None,
        "std": data["std"] if "std" in data else None,
        "pca_mean": data["pca_mean"] if "pca_mean" in data else None,
        "pca_components": data["pca_components"] if "pca_components" in data else None,
        "weighted": bool(data["weighted"]) if "weighted" in data else True,
    }


def load_mlp_model():
    path = MODEL_DIR / "mlp.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "w1": data["w1"],
        "b1": data["b1"],
        "w2": data["w2"],
        "b2": data["b2"],
    }


def predict_knn(model, sample: np.ndarray) -> tuple[int, float]:
    data_train = model["data_train"]
    labels_train = model["labels_train"]
    k = model["k"]
    mean = model.get("mean")
    std = model.get("std")
    pca_mean = model.get("pca_mean")
    pca_components = model.get("pca_components")
    weighted = model.get("weighted", True)
    if mean is not None and std is not None:
        sample = (sample / 255.0 - mean) / std
    elif data_train.max() > 1.5:
        sample = sample / 255.0
    if pca_mean is not None and pca_components is not None:
        sample = (sample - pca_mean) @ pca_components
    distances = np.linalg.norm(data_train - sample, axis=1)
    nearest = np.argsort(distances)[:k]
    votes = labels_train[nearest]
    if weighted:
        weights = 1.0 / (distances[nearest] + 1e-8)
        counts = np.bincount(votes, weights=weights, minlength=len(CIFAR_LABELS))
        label_idx = int(np.argmax(counts))
        confidence = float(counts[label_idx] / np.sum(weights)) if weights.size else 0.0
    else:
        counts = np.bincount(votes, minlength=len(CIFAR_LABELS))
        label_idx = int(np.argmax(counts))
        confidence = float(counts[label_idx] / k) if k > 0 else 0.0
    return label_idx, confidence


def predict_mlp(model, sample: np.ndarray) -> tuple[int, float]:
    w1 = model["w1"]
    b1 = model["b1"]
    w2 = model["w2"]
    b2 = model["b2"]

    z1 = sample @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    a2 = softmax(z2)
    label_idx = int(np.argmax(a2, axis=1)[0])
    confidence = float(a2[0, label_idx])
    return label_idx, confidence


def error_response(message: str, code: int = 501):
    return jsonify({"error": message}), code


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path: str):
    return send_from_directory(APP_DIR, path)


@app.get("/uploads")
def list_uploads():
    return jsonify({"items": load_upload_index()})


@app.post("/upload")
def save_upload():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return error_response("missing_image", 400)
    label = payload.get("label")
    label = label.strip() if isinstance(label, str) and label.strip() else None

    image, resized, image_format = decode_image_raw(payload["image"])
    timestamp = int(time.time() * 1000)
    ext = "jpg" if image_format.upper() in {"JPEG", "JPG"} else "png"
    filename = f"{timestamp}.{ext}"
    save_path = UPLOAD_DIR / filename
    if ext == "jpg":
        image.save(save_path, format="JPEG", quality=95)
    else:
        image.save(save_path, format="PNG")

    items = load_upload_index()
    items.insert(
        0,
        {
            "filename": filename,
            "label": label,
            "resized": resized,
            "created_at": timestamp,
        },
    )
    save_upload_index(items)
    return jsonify({"ok": True, "item": items[0]})


@app.get("/uploaded_images/<path:filename>")
def serve_uploaded_image(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.delete("/uploads/<path:filename>")
def delete_upload(filename: str):
    items = load_upload_index()
    remaining = [item for item in items if item.get("filename") != filename]
    if len(remaining) == len(items):
        return error_response("not_found", 404)
    save_upload_index(remaining)
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
    return jsonify({"ok": True, "filename": filename})


@app.post("/predict/knn")
def predict_knn_endpoint():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return error_response("missing_image", 400)

    model = load_knn_model()
    if model is None:
        return error_response("knn_model_not_available")

    sample, resized = decode_image(payload["image"])
    label_idx, confidence = predict_knn(model, sample)
    return jsonify(
        {"label": CIFAR_LABELS[label_idx], "confidence": confidence, "resized": resized}
    )


@app.post("/predict/mlp")
def predict_mlp_endpoint():
    payload = request.get_json(silent=True)
    if not payload or "image" not in payload:
        return error_response("missing_image", 400)

    model = load_mlp_model()
    if model is None:
        return error_response("mlp_model_not_available")

    sample, resized = decode_image(payload["image"])
    label_idx, confidence = predict_mlp(model, sample)
    return jsonify(
        {"label": CIFAR_LABELS[label_idx], "confidence": confidence, "resized": resized}
    )


def compute_pca(data: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0, keepdims=True)
    centered = data - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    reduced = centered @ components
    return reduced, mean, components


def knn_predict_batch(
    data_train: np.ndarray,
    labels_train: np.ndarray,
    data_query: np.ndarray,
    k: int,
    weighted: bool,
) -> np.ndarray:
    distances = np.linalg.norm(data_train[None, :, :] - data_query[:, None, :], axis=2)
    nearest = np.argsort(distances, axis=1)[:, :k]
    preds = []
    for i, neighbors in enumerate(nearest):
        votes = labels_train[neighbors]
        if weighted:
            weights = 1.0 / (distances[i, neighbors] + 1e-8)
            counts = np.bincount(votes, weights=weights, minlength=len(CIFAR_LABELS))
        else:
            counts = np.bincount(votes, minlength=len(CIFAR_LABELS))
        preds.append(int(np.argmax(counts)))
    return np.array(preds, dtype=np.int64)


def select_knn_k(
    data_train: np.ndarray,
    labels_train: np.ndarray,
    k_grid: list[int],
    val_samples: int,
    weighted: bool,
) -> int:
    rng = np.random.default_rng(42)
    n_samples = data_train.shape[0]
    if n_samples < 2:
        return k_grid[0]
    val_samples = min(val_samples, n_samples - 1)
    indices = rng.permutation(n_samples)
    val_idx = indices[:val_samples]
    train_idx = indices[val_samples:]
    train_data = data_train[train_idx]
    train_labels = labels_train[train_idx]
    val_data = data_train[val_idx]
    val_labels = labels_train[val_idx]
    best_k = k_grid[0]
    best_acc = -1.0
    for k in k_grid:
        k = min(k, train_data.shape[0])
        preds = knn_predict_batch(train_data, train_labels, val_data, k, weighted)
        acc = float(np.mean(preds == val_labels))
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k


def load_extra_images(dir_path: Path, label_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Extra image dir not found: {dir_path}")
    samples = []
    labels = []
    for path in sorted(dir_path.iterdir()):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = ImageOps.fit(image, (32, 32), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32).reshape(1, -1)
        except OSError:
            continue
        samples.append(array)
        labels.append(label_idx)
    if not samples:
        raise ValueError(f"No readable images found in {dir_path}")
    data = np.vstack(samples)
    labels = np.array(labels, dtype=np.int64)
    return data, labels


def train_knn(
    data_train,
    labels_train,
    k: int,
    out_path: Path,
    pca_components: int,
    auto_k: bool,
    k_grid: list[int],
    val_samples: int,
    weighted: bool,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_train = data_train / 255.0
    mean = data_train.mean(axis=0, keepdims=True)
    std = data_train.std(axis=0, keepdims=True) + 1e-6
    data_train = (data_train - mean) / std
    pca_mean = None
    pca_components_matrix = None
    if pca_components and pca_components > 0:
        data_train, pca_mean, pca_components_matrix = compute_pca(data_train, pca_components)
    if auto_k:
        k = select_knn_k(data_train, labels_train, k_grid, val_samples, weighted)
    np.savez(
        out_path,
        data_train=data_train,
        labels_train=labels_train,
        k=k,
        mean=mean,
        std=std,
        pca_mean=pca_mean,
        pca_components=pca_components_matrix,
        weighted=int(weighted),
    )


def train_mlp(data_train, labels_train, d_h: int, learning_rate: float, epochs: int, out_path: Path):
    d_in = data_train.shape[1]
    d_out = int(np.max(labels_train) + 1)

    rng = np.random.default_rng()
    w1 = 2 * rng.random((d_in, d_h)) - 1
    b1 = np.zeros((1, d_h))
    w2 = 2 * rng.random((d_h, d_out)) - 1
    b2 = np.zeros((1, d_out))

    onehot = np.zeros((labels_train.shape[0], d_out))
    onehot[np.arange(labels_train.shape[0]), labels_train] = 1

    for _ in range(epochs):
        z1 = data_train @ w1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ w2 + b2
        a2 = softmax(z2)

        dC_dz2 = (a2 - onehot) / labels_train.shape[0]
        dC_dw2 = a1.T @ dC_dz2
        dC_db2 = np.sum(dC_dz2, axis=0, keepdims=True)

        dC_da1 = dC_dz2 @ w2.T
        dC_dz1 = dC_da1 * sigmoid_deriv(a1)
        dC_dw1 = data_train.T @ dC_dz1
        dC_db1 = np.sum(dC_dz1, axis=0, keepdims=True)

        w1 -= learning_rate * dC_dw1
        b1 -= learning_rate * dC_db1
        w2 -= learning_rate * dC_dw2
        b2 -= learning_rate * dC_db2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, w1=w1, b1=b1, w2=w2, b2=b2)


def run_training(args: argparse.Namespace):
    data, labels = read_cifar(str(APP_DIR / "data" / "cifar-10-batches-py"))
    data_train, labels_train, _, _ = split_dataset(data, labels, split=0.9)

    if args.knn_samples:
        data_train = data_train[: args.knn_samples]
        labels_train = labels_train[: args.knn_samples]
    if args.knn_extra_dir:
        extra_data, extra_labels = load_extra_images(
            Path(args.knn_extra_dir), LABEL_TO_INDEX[args.knn_extra_label]
        )
        data_train = np.vstack([data_train, extra_data])
        labels_train = np.concatenate([labels_train, extra_labels])
    rng = np.random.default_rng(123)
    perm = rng.permutation(data_train.shape[0])
    data_train = data_train[perm]
    labels_train = labels_train[perm]
    train_knn(
        data_train,
        labels_train,
        args.k,
        MODEL_DIR / "knn.npz",
        args.knn_pca,
        args.knn_auto_k,
        args.knn_k_grid,
        args.knn_val_samples,
        args.knn_weighted,
    )

    if args.mlp_samples:
        data_train_mlp = data_train[: args.mlp_samples]
        labels_train_mlp = labels_train[: args.mlp_samples]
    else:
        data_train_mlp = data_train
        labels_train_mlp = labels_train

    train_mlp(
        data_train_mlp,
        labels_train_mlp,
        args.hidden,
        args.learning_rate,
        args.epochs,
        MODEL_DIR / "mlp.npz",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the KNN/MLP demo site and API.")
    parser.add_argument("--train", action="store_true", help="Train and save models before serving.")
    parser.add_argument("--knn-samples", type=int, default=3000)
    parser.add_argument("--mlp-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--knn-pca", type=int, default=128)
    parser.add_argument("--knn-auto-k", action="store_true", help="Auto-tune K using validation.")
    parser.add_argument(
        "--knn-k-grid",
        type=int,
        nargs="+",
        default=[1, 3, 5, 7, 9, 11],
        help="Candidate K values when using --knn-auto-k.",
    )
    parser.add_argument("--knn-val-samples", type=int, default=600)
    parser.add_argument("--knn-weighted", action="store_true", help="Use distance-weighted voting.")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train models and exit without starting the server.",
    )
    parser.add_argument(
        "--knn-extra-dir",
        type=str,
        default=None,
        help="Optional directory of extra images to include in KNN training.",
    )
    parser.add_argument(
        "--knn-extra-label",
        type=str,
        default="cat",
        choices=sorted(LABEL_TO_INDEX.keys()),
        help="Label for images in --knn-extra-dir.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    if args.train:
        run_training(args)

    if args.train_only:
        raise SystemExit(0)

    app.run(host=args.host, port=args.port, debug=True)
