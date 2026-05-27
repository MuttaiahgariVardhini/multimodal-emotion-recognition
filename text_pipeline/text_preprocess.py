import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import accuracy_score

from speech_model import SpeechEmotionModel


# =====================================
# LOAD PREPROCESSED DATA
# =====================================

X_train = np.load(
    "models/speech_pipeline/X_train.npy"
)

X_test = np.load(
    "models/speech_pipeline/X_test.npy"
)

y_train = np.load(
    "models/speech_pipeline/y_train.npy"
)

y_test = np.load(
    "models/speech_pipeline/y_test.npy"
)

print("Training Feature Shape:", X_train.shape)

print("Testing Feature Shape:", X_test.shape)


# =====================================
# CONVERT TO TENSORS
# =====================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# =====================================
# DATA LOADER
# =====================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
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
# INITIALIZE MODEL
# =====================================

model = SpeechEmotionModel(
    num_classes=7
).to(device)


# =====================================
# LOSS FUNCTION + OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================================
# TRAINING LOOP
# =====================================

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    for features, labels in train_loader:

        features = features.to(device)

        labels = labels.to(device)

        # Forward Pass
        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Predictions
        _, predicted = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# =====================================
# SAVE TRAINED MODEL
# =====================================

torch.save(
    model.state_dict(),
    "models/speech_pipeline/speech_emotion_model.pth"
)

print(
    "\nSpeech model saved successfully."
)