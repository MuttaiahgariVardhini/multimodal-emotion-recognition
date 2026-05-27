import numpy as np

import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from model import FusionEmotionModel


# =====================================
# LOAD FUSION FEATURES
# =====================================

X = np.load(
    "models/fusion_pipeline/fusion_X.npy"
)

y = np.load(
    "models/fusion_pipeline/fusion_y.npy"
)

print("Fusion Feature Shape:", X.shape)

print("Labels Shape:", y.shape)


# =====================================
# CONVERT TO TENSORS
# =====================================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.long
)


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device:", device)


# =====================================
# LOAD MODEL
# =====================================

model = FusionEmotionModel()

model.load_state_dict(
    torch.load(
        "models/fusion_pipeline/fusion_emotion_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nFusion model loaded successfully.")


# =====================================
# PREDICTIONS
# =====================================

with torch.no_grad():

    X_tensor = X_tensor.to(device)

    speech_features = X_tensor[:, :256]

    text_features = X_tensor[:, 256:]

    outputs = model(
        speech_features,
        text_features
    )

    _, predictions = torch.max(
        outputs,
        1
    )

    predictions = predictions.cpu().numpy()


# =====================================
# ACCURACY
# =====================================

accuracy = accuracy_score(
    y,
    predictions
)

print(
    f"\nFusion Test Accuracy: {accuracy * 100:.2f}%"
)


# =====================================
# CLASSIFICATION REPORT
# =====================================

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

print("\nClassification Report:\n")

print(
    classification_report(
        y,
        predictions,
        target_names=emotion_labels
    )
)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y,
    predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=emotion_labels,
    yticklabels=emotion_labels
)

plt.title(
    "Fusion Emotion Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    "Results/plots/fusion_confusion_matrix.png"
)

print(
    "\nFusion confusion matrix saved successfully."
)

plt.close()


# =====================================
# t-SNE VISUALIZATION
# =====================================

print(
    "\nRunning Fusion t-SNE..."
)

sample_features = X[:300]

sample_labels = y[:300]

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=5
)

features_2d = tsne.fit_transform(
    sample_features
)

print(
    "Fusion t-SNE completed successfully."
)


# =====================================
# PLOT t-SNE
# =====================================

plt.figure(figsize=(10, 8))

sns.scatterplot(
    x=features_2d[:, 0],
    y=features_2d[:, 1],
    hue=[
        emotion_labels[label]
        for label in sample_labels
    ],
    palette='deep'
)

plt.title(
    "Fusion t-SNE Visualization"
)

plt.xlabel(
    "t-SNE Component 1"
)

plt.ylabel(
    "t-SNE Component 2"
)

plt.legend(title="Emotion")

plt.tight_layout()

plt.savefig(
    "Results/plots/fusion_tsne.png"
)

print(
    "Fusion t-SNE plot saved successfully."
)

plt.close()

print(
    "\nFusion pipeline evaluation completed successfully."
)