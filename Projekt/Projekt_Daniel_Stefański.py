from __future__ import annotations

import argparse
import json
import os
import random
import platform
import sys
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LAB5_KERAS_CACHE = PROJECT_DIR.parent / "Lab5" / ".keras_cache"
if LAB5_KERAS_CACHE.exists():
    os.environ.setdefault("KERAS_HOME", str(LAB5_KERAS_CACHE))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SEED = 42
IMG_SIZE = 160
BATCH_SIZE = 32
OUTPUT_DIR = PROJECT_DIR / "wyniki"
TERMINAL_DIR = PROJECT_DIR / "terminal"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_terminal_logging() -> tuple[object, object, list[object], Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = TERMINAL_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        latest_log_path = log_dir / "latest_run_log.txt"
        dated_log_path = log_dir / f"run_log_{timestamp}.txt"
        latest_log = latest_log_path.open("w", encoding="utf-8")
        dated_log = dated_log_path.open("w", encoding="utf-8")
    except PermissionError:
        log_dir = OUTPUT_DIR / "logi"
        log_dir.mkdir(parents=True, exist_ok=True)
        latest_log_path = log_dir / "latest_run_log.txt"
        dated_log_path = log_dir / f"run_log_{timestamp}.txt"
        latest_log = latest_log_path.open("w", encoding="utf-8")
        dated_log = dated_log_path.open("w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee = TeeStream(original_stdout, latest_log, dated_log)
    sys.stdout = tee
    sys.stderr = tee
    return original_stdout, original_stderr, [latest_log, dated_log], latest_log_path, dated_log_path


def close_terminal_logging(original_stdout: object, original_stderr: object, log_files: list[object]) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    for log_file in log_files:
        log_file.close()


@dataclass
class ExperimentConfig:
    name: str
    learning_rate: float = 1e-4
    use_augmentation: bool = True
    fine_tune: bool = False
    dense_units: int = 128
    dropout: float = 0.3
    epochs: int = 6
    fine_tune_epochs: int = 4


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_keras_cache() -> None:
    if LAB5_KERAS_CACHE.exists():
        os.environ.setdefault("KERAS_HOME", str(LAB5_KERAS_CACHE))


def collect_run_info(quick: bool, data_dir: Path | None = None) -> dict[str, object]:
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    return {
        "run_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quick_mode": quick,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "gpu_available": len(gpus) > 0,
        "gpu_devices": [device.name for device in gpus],
        "cpu_devices": [device.name for device in cpus],
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "dataset_dir": str(data_dir) if data_dir is not None else None,
    }


def save_run_info(run_info: dict[str, object]) -> None:
    with (OUTPUT_DIR / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print("\n=== Informacje o uruchomieniu ===")
    print(f"Data uruchomienia: {run_info['run_datetime']}")
    print(f"TensorFlow: {run_info['tensorflow_version']}")
    print(f"Tryb szybki: {run_info['quick_mode']}")
    if run_info["gpu_available"]:
        print("Wykryte GPU:")
        for device_name in run_info["gpu_devices"]:
            print(f"  - {device_name}")
    else:
        print("Wykryte GPU: brak, trening zostanie wykonany na CPU.")


def find_dataset_dir(custom_path: str | None = None) -> Path:
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend(
        [
            PROJECT_DIR / "flower_photos",
            PROJECT_DIR.parent / "Lab5" / ".keras_cache" / "datasets" / "flower_photos" / "flower_photos",
            PROJECT_DIR.parent / "Lab6" / "flower_photos",
        ]
    )

    for path in candidates:
        if path.exists() and any(p.is_dir() for p in path.iterdir()):
            return path
    raise FileNotFoundError("Nie znaleziono katalogu flower_photos. Podaj sciezke przez --data-dir.")


def collect_image_paths(data_dir: Path) -> tuple[list[str], list[int], list[str]]:
    class_names = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    image_paths: list[str] = []
    labels: list[int] = []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for image_path in class_dir.glob(ext):
                image_paths.append(str(image_path))
                labels.append(label)

    if not image_paths:
        raise ValueError(f"Brak obrazow w katalogu: {data_dir}")

    return image_paths, labels, class_names


def split_dataset(paths: list[str], labels: list[int]) -> dict[str, tuple[list[str], list[int]]]:
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="augmentation",
    )


def build_model(config: ExperimentConfig, class_count: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    if config.use_augmentation:
        x = build_augmentation()(x)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config.dropout)(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(config.dropout)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name=f"mobilenetv2_{config.name}")
    compile_model(model, config.learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def plot_history(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title)

    axes[0].plot(history.get("accuracy", []), label="train")
    axes[0].plot(history.get("val_accuracy", []), label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history.get("loss", []), label="train")
    axes[1].plot(history.get("val_loss", []), label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Klasa prawdziwa")
    ax.set_title("Macierz pomylek")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def predict_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    probabilities: list[np.ndarray] = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        probabilities.extend(probs)
        y_true.extend(labels.numpy().tolist())

    probs_array = np.asarray(probabilities)
    y_pred = np.argmax(probs_array, axis=1)
    return np.asarray(y_true), y_pred, probs_array


def save_wrong_predictions(
    model: tf.keras.Model,
    test_paths: list[str],
    test_labels: list[int],
    class_names: list[str],
    output_path: Path,
    limit: int = 12,
) -> list[dict[str, object]]:
    wrong: list[dict[str, object]] = []
    sample_dataset = make_dataset(test_paths, test_labels, training=False)
    y_true, y_pred, probs = predict_dataset(model, sample_dataset)

    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label != pred_label:
            wrong.append(
                {
                    "path": test_paths[idx],
                    "true": class_names[int(true_label)],
                    "predicted": class_names[int(pred_label)],
                    "confidence": float(np.max(probs[idx])),
                }
            )

    selected = wrong[:limit]
    if selected:
        cols = 4
        rows = int(np.ceil(len(selected) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.3 * rows))
        axes = np.asarray(axes).reshape(-1)
        for ax, item in zip(axes, selected):
            image = tf.keras.utils.load_img(item["path"], target_size=(IMG_SIZE, IMG_SIZE))
            ax.imshow(image)
            ax.set_title(
                f"true: {item['true']}\npred: {item['predicted']} ({item['confidence']:.2f})",
                fontsize=9,
            )
            ax.axis("off")
        for ax in axes[len(selected) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    return wrong


def train_experiment(
    config: ExperimentConfig,
    datasets: dict[str, tf.data.Dataset],
    class_names: list[str],
    split_paths: dict[str, tuple[list[str], list[int]]],
    quick: bool = False,
) -> dict[str, object]:
    exp_dir = OUTPUT_DIR / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    train_epochs = min(config.epochs, 2) if quick else config.epochs
    fine_tune_epochs = min(config.fine_tune_epochs, 1) if quick else config.fine_tune_epochs

    model = build_model(config, class_count=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=2, min_lr=1e-6),
    ]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["val"],
        epochs=train_epochs,
        callbacks=callbacks,
        verbose=1,
    )
    merged_history = {key: list(value) for key, value in history.history.items()}

    if config.fine_tune:
        base_model = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name)
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        compile_model(model, learning_rate=config.learning_rate / 10)
        ft_history = model.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )
        for key, value in ft_history.history.items():
            merged_history.setdefault(key, []).extend(list(value))

    test_loss, test_accuracy = model.evaluate(datasets["test"], verbose=0)
    y_true, y_pred, probabilities = predict_dataset(model, datasets["test"])
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    plot_history(merged_history, exp_dir / "history.png", config.name)
    plot_confusion_matrix(cm, class_names, exp_dir / "confusion_matrix.png")
    wrong = save_wrong_predictions(
        model,
        split_paths["test"][0],
        split_paths["test"][1],
        class_names,
        exp_dir / "wrong_predictions.png",
    )

    best_val_acc = max(merged_history.get("val_accuracy", [0.0]))
    result = {
        "config": asdict(config),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_val_accuracy": float(best_val_acc),
        "epochs_ran": len(merged_history.get("loss", [])),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "wrong_predictions_count": len(wrong),
        "history_path": str(exp_dir / "history.png"),
        "confusion_matrix_path": str(exp_dir / "confusion_matrix.png"),
        "wrong_predictions_path": str(exp_dir / "wrong_predictions.png"),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
    }

    with (exp_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    model.save(exp_dir / "model.keras")
    return result


def save_dataset_summary(split_paths: dict[str, tuple[list[str], list[int]]], class_names: list[str]) -> dict[str, object]:
    summary: dict[str, object] = {"class_names": class_names, "splits": {}}
    for split_name, (_, labels) in split_paths.items():
        counts = {class_name: 0 for class_name in class_names}
        for label in labels:
            counts[class_names[label]] += 1
        summary["splits"][split_name] = {"total": len(labels), "class_counts": counts}

    with (OUTPUT_DIR / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def make_experiments(quick: bool) -> list[ExperimentConfig]:
    base_epochs = 2 if quick else 6
    ft_epochs = 1 if quick else 4
    return [
        ExperimentConfig(
            name="exp1_lr_1e-4",
            learning_rate=1e-4,
            use_augmentation=True,
            fine_tune=False,
            epochs=base_epochs,
        ),
        ExperimentConfig(
            name="exp1_lr_1e-3",
            learning_rate=1e-3,
            use_augmentation=True,
            fine_tune=False,
            epochs=base_epochs,
        ),
        ExperimentConfig(
            name="exp2_without_augmentation",
            learning_rate=1e-4,
            use_augmentation=False,
            fine_tune=False,
            epochs=base_epochs,
        ),
        ExperimentConfig(
            name="exp3_fine_tuning",
            learning_rate=1e-4,
            use_augmentation=True,
            fine_tune=True,
            epochs=base_epochs,
            fine_tune_epochs=ft_epochs,
        ),
    ]


def save_comparison(results: list[dict[str, object]]) -> None:
    rows = []
    for result in results:
        config = result["config"]
        rows.append(
            {
                "name": config["name"],
                "learning_rate": config["learning_rate"],
                "augmentation": config["use_augmentation"],
                "fine_tune": config["fine_tune"],
                "best_val_accuracy": result["best_val_accuracy"],
                "test_accuracy": result["test_accuracy"],
                "test_loss": result["test_loss"],
                "wrong_predictions_count": result["wrong_predictions_count"],
                "mean_confidence": result["mean_confidence"],
            }
        )

    with (OUTPUT_DIR / "comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    labels = [row["name"] for row in rows]
    test_acc = [row["test_accuracy"] for row in rows]
    val_acc = [row["best_val_accuracy"] for row in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, val_acc, width, label="best val accuracy")
    ax.bar(x + width / 2, test_acc, width, label="test accuracy")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Porownanie eksperymentow")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Projekt: klasyfikacja i analiza obrazow kwiatow.")
    parser.add_argument("--data-dir", default=None, help="Opcjonalna sciezka do katalogu flower_photos.")
    parser.add_argument("--quick", action="store_true", help="Krotki tryb testowy z mala liczba epok.")
    args = parser.parse_args()

    original_stdout, original_stderr, log_files, latest_log_path, dated_log_path = setup_terminal_logging()
    try:
        run_project(args, latest_log_path, dated_log_path)
    finally:
        close_terminal_logging(original_stdout, original_stderr, log_files)


def run_project(args: argparse.Namespace, latest_log_path: Path, dated_log_path: Path) -> None:
    print("Log terminala:")
    print(f"  aktualny: {latest_log_path}")
    print(f"  archiwum: {dated_log_path}")

    configure_keras_cache()
    set_reproducibility()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = find_dataset_dir(args.data_dir)
    run_info = collect_run_info(quick=args.quick, data_dir=data_dir)
    save_run_info(run_info)

    paths, labels, class_names = collect_image_paths(data_dir)
    split_paths = split_dataset(paths, labels)
    datasets = {
        split: make_dataset(split_paths[split][0], split_paths[split][1], training=(split == "train"))
        for split in ("train", "val", "test")
    }

    summary = save_dataset_summary(split_paths, class_names)
    print("Dataset:", data_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    results = []
    for config in make_experiments(args.quick):
        print(f"\n=== Start eksperymentu: {config.name} ===")
        results.append(train_experiment(config, datasets, class_names, split_paths, quick=args.quick))

    save_comparison(results)
    print(f"\nGotowe. Wyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
