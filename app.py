# -*- coding: utf-8 -*-
"""
CNN image-classification demo for Streamlit.

Run locally:
    pip install -r requirements.txt
    streamlit run untitled10_fixed.py

This version does not require matplotlib. Streamlit is used for charts and
image display, so it also avoids the ModuleNotFoundError shown in the screenshot.
"""

import numpy as np
import pandas as pd
import streamlit as st



# ----------------------------- Page setup -----------------------------
st.set_page_config(
    page_title="CNN Image Classification",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 CNN Image Classification")
st.write(
    "This demo generates a synthetic dataset of circles, squares, and triangles, "
    "then trains and evaluates a Convolutional Neural Network (CNN)."
)

CLASS_NAMES = ["Circle", "Square", "Triangle"]
IMAGE_SIZE = 32


def stratified_split(X, y, test_size=0.2, random_state=42):
    """Split arrays while preserving the proportion of each class."""
    rng = np.random.default_rng(random_state)
    train_indices, test_indices = [], []
    for class_id in np.unique(y):
        class_indices = np.flatnonzero(y == class_id)
        class_indices = rng.permutation(class_indices)
        n_test = max(1, int(round(len(class_indices) * test_size)))
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])
    train_indices = rng.permutation(np.asarray(train_indices, dtype=np.int64))
    test_indices = rng.permutation(np.asarray(test_indices, dtype=np.int64))
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def make_classification_report(y_true, y_pred, class_names):
    """Return a sklearn-style report and metrics using NumPy only."""
    rows = {}
    supports, precisions, recalls, f1_scores = [], [], [], []
    for class_id, name in enumerate(class_names):
        tp = int(np.sum((y_true == class_id) & (y_pred == class_id)))
        fp = int(np.sum((y_true != class_id) & (y_pred == class_id)))
        fn = int(np.sum((y_true == class_id) & (y_pred != class_id)))
        support = int(np.sum(y_true == class_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    total = int(sum(supports))
    accuracy = float(np.mean(y_true == y_pred)) if total else 0.0
    rows["accuracy"] = accuracy
    rows["macro avg"] = {
        "precision": float(np.mean(precisions)), "recall": float(np.mean(recalls)),
        "f1-score": float(np.mean(f1_scores)), "support": total,
    }
    weights = np.asarray(supports, dtype=float)
    rows["weighted avg"] = {
        "precision": float(np.average(precisions, weights=weights)) if total else 0.0,
        "recall": float(np.average(recalls, weights=weights)) if total else 0.0,
        "f1-score": float(np.average(f1_scores, weights=weights)) if total else 0.0,
        "support": total,
    }

    header = f"{'':<14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines = [header, ""]
    for name in class_names:
        r = rows[name]
        lines.append(f"{name:<14}{r['precision']:>10.2f}{r['recall']:>10.2f}{r['f1-score']:>10.2f}{r['support']:>10d}")
    lines.extend(["", f"{'accuracy':<34}{accuracy:>10.2f}{total:>10d}"])
    for name in ("macro avg", "weighted avg"):
        r = rows[name]
        lines.append(f"{name:<14}{r['precision']:>10.2f}{r['recall']:>10.2f}{r['f1-score']:>10.2f}{r['support']:>10d}")
    return rows, "\n".join(lines)


# -------------------------- Dataset generation -------------------------
@st.cache_data
def generate_shapes_dataset(num_samples=900, image_size=IMAGE_SIZE):
    """Create grayscale images containing circles, squares, and triangles."""
    rng = np.random.default_rng(42)
    images = np.zeros((num_samples, image_size, image_size), dtype=np.float32)
    labels = np.arange(num_samples, dtype=np.int64) % 3

    rows, cols = np.ogrid[:image_size, :image_size]
    center = image_size // 2

    for i, label in enumerate(labels):
        img = rng.uniform(0.0, 0.1, (image_size, image_size)).astype(np.float32)

        if label == 0:
            radius = int(rng.integers(6, 12))
            mask = (rows - center) ** 2 + (cols - center) ** 2 < radius ** 2
            img[mask] += 0.8

        elif label == 1:
            side = int(rng.integers(10, 20))
            start = (image_size - side) // 2
            img[start:start + side, start:start + side] += 0.8

        else:
            height = int(rng.integers(10, 20))
            start_row = (image_size - height) // 2
            for row_idx in range(height):
                width = max(1, int(row_idx * (height / max(height - 1, 1))))
                start_col = (image_size - width) // 2
                img[start_row + row_idx, start_col:start_col + width] += 0.8

        images[i] = np.clip(img, 0.0, 1.0)

    return images, labels


# ----------------------------- CNN training ----------------------------
@st.cache_resource(show_spinner=False)
def train_cnn():
    """Train the CNN once and cache the model/results across Streamlit reruns."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)

    X, y = generate_shapes_dataset()
    X = np.expand_dims(X, axis=-1)

    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=0.2, random_state=42
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_split=0.1,
        batch_size=32,
        verbose=0,
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(y_test == predictions))
    report_dict, report_text = make_classification_report(
        y_test, predictions, CLASS_NAMES
    )

    comparison = pd.DataFrame({
        "Actual Label": [CLASS_NAMES[int(v)] for v in y_test],
        "Predicted Label": [CLASS_NAMES[int(v)] for v in predictions],
    })
    comparison["Match"] = comparison["Actual Label"] == comparison["Predicted Label"]

    return (
        model,
        X,
        y,
        history.history,
        accuracy,
        report_dict,
        report_text,
        comparison,
    )


# ------------------------------- App UI --------------------------------
with st.spinner("Generating dataset and training the CNN. This may take a few minutes the first time..."):
    try:
        (
            model,
            X,
            y,
            history,
            accuracy,
            report_dict,
            report_text,
            comparison,
        ) = train_cnn()
    except ModuleNotFoundError as exc:
        st.error(
            f"Missing Python package: {exc.name}. "
            "Add `tensorflow-cpu` to requirements.txt, commit/push the file, "
            "then reboot the app on Streamlit Cloud. Locally run: "
            "`pip install -r requirements.txt`."
        )
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.success("Dataset generated and CNN training completed.")

# Dataset information
st.subheader("1. Dataset overview")
counts = pd.Series(y).value_counts().sort_index()
distribution = pd.DataFrame({
    "Shape": CLASS_NAMES,
    "Number of images": [int(counts.get(i, 0)) for i in range(3)],
}).set_index("Shape")

col1, col2, col3 = st.columns(3)
col1.metric("Total images", len(y))
col2.metric("Image size", f"{IMAGE_SIZE} × {IMAGE_SIZE}")
col3.metric("Number of classes", len(CLASS_NAMES))

left, right = st.columns(2)
with left:
    st.write("Class distribution")
    st.bar_chart(distribution)

with right:
    st.write("Example images")
    sample_cols = st.columns(3)
    for class_id, col in enumerate(sample_cols):
        sample_index = int(np.where(y == class_id)[0][0])
        col.image(
            X[sample_index].squeeze(),
            caption=CLASS_NAMES[class_id],
            clamp=True,
            width="stretch",
        )

# Training history
st.subheader("2. Model training")
history_df = pd.DataFrame({
    "Epoch": np.arange(1, len(history["accuracy"]) + 1),
    "Training accuracy": history["accuracy"],
    "Validation accuracy": history["val_accuracy"],
}).set_index("Epoch")
st.line_chart(history_df)

# Evaluation
st.subheader("3. Model evaluation")
st.metric("Test accuracy", f"{accuracy * 100:.2f}%")
st.text("Classification report:")
st.code(report_text)

# Prediction comparison
st.subheader("4. Actual vs predicted labels")
st.dataframe(comparison.head(25), use_container_width=True)

st.caption(
    "Note: This app currently trains on a generated shapes dataset. "
    "It detects ZIP files only if you add dataset-loading code; it does not "
    "train on archive (1).zip or archive (2).zip."
)
