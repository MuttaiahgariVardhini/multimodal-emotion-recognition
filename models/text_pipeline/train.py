import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from collections import Counter

from text_model import TextEmotionModel


# =====================================
# LOAD DATA
# =====================================

train_df = pd.read_csv(
    "models/text_pipeline/train.csv"
)

test_df = pd.read_csv(
    "models/text_pipeline/test.csv"
)

print(train_df.head())


# =====================================
# TOKENIZATION
# =====================================

train_texts = train_df["text"].tolist()

test_texts = test_df["text"].tolist()


# =====================================
# BUILD VOCABULARY
# =====================================

all_words = []

for text in train_texts:

    all_words.append(
        text.lower()
    )


word_counts = Counter(
    all_words
)

vocab = {
    word: idx + 1
    for idx, (word, count)
    in enumerate(word_counts.items())
}

# Padding token
vocab["<PAD>"] = 0

print(
    "\nVocabulary Size:",
    len(vocab)
)


# =====================================
# TEXT TO SEQUENCE
# =====================================

def text_to_sequence(text):

    return [
        vocab.get(
            text.lower(),
            0
        )
    ]


X_train = [
    text_to_sequence(text)
    for text in train_texts
]

X_test = [
    text_to_sequence(text)
    for text in test_texts
]


# =====================================
# CONVERT TO NUMPY ARRAYS
# =====================================

X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = train_df["label"].values

y_test = test_df["label"].values


print(
    "\nTraining Shape:",
    X_train.shape
)

print(
    "Testing Shape:",
    X_test.shape
)


# =====================================
# CONVERT TO TENSORS
# =====================================

X_train = torch.tensor(
    X_train,
    dtype=torch.long
)

X_test = torch.tensor(
    X_test,
    dtype=torch.long
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# =====================================
# DATALOADERS
# =====================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

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

print(
    "\nUsing Device:",
    device
)


# =====================================
# LOAD MODEL
# =====================================

model = TextEmotionModel(
    vocab_size=len(vocab)
)

model = model.to(device)


# =====================================
# LOSS + OPTIMIZER
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

    correct = 0

    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += (
            loss.item() * inputs.size(0)
        )

        # Accuracy
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
    "models/text_pipeline/text_emotion_model.pth"
)

print(
    "\nText model saved successfully."
)