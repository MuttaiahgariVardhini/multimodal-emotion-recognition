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

from speech_model import SpeechEmotionModel


# =====================================
# LOAD TEST DATA
# =====================================

X_test = np.load(
    "models/speech_pipeline/X_test.npy"
)

y_test = np.load(
    "models/speech_pipeline/y_test.npy"
)

print("Test Data Shape:", X_test.shape)


# =====================================
# CONVERT TO TENSORS
# =====================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# =====================================
# DEVICE CONFIGURATION
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device:", device)


# =====================================
# LOAD TRAINED MODEL
# =====================================

model = SpeechEmotionModel(
    num_classes=7
)

model.load_state_dict(
    torch.load(
        "models/speech_pipeline/speech_emotion_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("\nModel loaded successfully.")


# =====================================
# PREDICTIONS
# =====================================

with torch.no_grad():

    X_test_tensor = X_test_tensor.to(device)

    outputs = model(X_test_tensor)

    _, predictions = torch.max(
        outputs,
        1
    )

    predictions = predictions.cpu().numpy()


# =====================================
# ACCURACY
# =====================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
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
        y_test,
        predictions,
        target_names=emotion_labels
    )
)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
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
    "Speech Emotion Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

# SAVE CONFUSION MATRIX
plt.savefig(
    "Results/plots/speech_confusion_matrix.png"
)

print(
    "\nSpeech confusion matrix saved successfully."
)

plt.close()


# =====================================
# EXTRACT EMBEDDINGS
# =====================================

print(
    "\nExtracting embeddings for t-SNE visualization..."
)

with torch.no_grad():

    X_test_tensor = X_test_tensor.to(device)

    # Convert shape:
    # (batch, 40, 150)
    # ->
    # (batch, 150, 40)

    x = X_test_tensor.permute(
        0,
        2,
        1
    )

    # LSTM output

    output, (hidden, cell) = model.lstm(x)

    hidden_forward = hidden[-2]

    hidden_backward = hidden[-1]

    hidden_combined = torch.cat(
        (
            hidden_forward,
            hidden_backward
        ),
        dim=1
    )

    embeddings = (
        hidden_combined
        .cpu()
        .numpy()
    )

print(
    "Embedding Shape:",
    embeddings.shape
)


# =====================================
# REDUCE SAMPLE SIZE
# =====================================

sample_embeddings = embeddings[:300]

sample_labels = y_test[:300]


# =====================================
# t-SNE
# =====================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=5
)

embeddings_2d = tsne.fit_transform(
    sample_embeddings
)

print("t-SNE completed successfully.")


# =====================================
# t-SNE VISUALIZATION
# =====================================

plt.figure(figsize=(10, 8))

sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=[
        emotion_labels[label]
        for label in sample_labels
    ],
    palette='deep'
)

plt.title(
    "t-SNE Visualization of Speech Emotion Embeddings"
)

plt.xlabel(
    "t-SNE Component 1"
)

plt.ylabel(
    "t-SNE Component 2"
)

plt.legend(title="Emotion")

plt.tight_layout()

# SAVE t-SNE PLOT
plt.savefig(
    "Results/plots/speech_tsne.png"
)

print(
    "Speech t-SNE plot saved successfully."
)

plt.close()

print(
    "\nSpeech pipeline evaluation completed successfully."
)