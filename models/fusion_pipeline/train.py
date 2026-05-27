import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

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
# TRAIN TEST SPLIT
# =====================================

dataset = TensorDataset(
    X_tensor,
    y_tensor
)

train_size = int(
    0.8 * len(dataset)
)

test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)


# =====================================
# DATALOADERS
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
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

model = model.to(device)


# =====================================
# LOSS + OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# =====================================
# TRAINING LOOP
# =====================================

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for features, labels in train_loader:

        features = features.to(device)

        labels = labels.to(device)

        # Split speech/text features

        speech_features = features[:, :256]

        text_features = features[:, 256:]

        optimizer.zero_grad()

        outputs = model(
            speech_features,
            text_features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# =====================================
# SAVE MODEL
# =====================================

torch.save(
    model.state_dict(),
    "models/fusion_pipeline/fusion_emotion_model.pth"
)

print(
    "\nFusion model saved successfully."
)