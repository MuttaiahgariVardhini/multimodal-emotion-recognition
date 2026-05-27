import os
import sys

import numpy as np
import pandas as pd

import torch

from collections import Counter


# =====================================
# ADD SPEECH/TEXT PIPELINE PATHS
# =====================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SPEECH_PATH = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "../speech_pipeline"
    )
)

TEXT_PATH = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "../text_pipeline"
    )
)

sys.path.insert(0, SPEECH_PATH)

from speech_model import SpeechEmotionModel

sys.path.insert(0, TEXT_PATH)

from text_model import TextEmotionModel


# =====================================
# DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device:", device)


# =====================================
# LOAD SPEECH DATA
# =====================================

X_speech = np.load(
    "models/speech_pipeline/X_test.npy"
)

y_speech = np.load(
    "models/speech_pipeline/y_test.npy"
)

print(
    "\nSpeech Data Shape:",
    X_speech.shape
)


# =====================================
# LOAD TEXT DATA
# =====================================

train_df = pd.read_csv(
    "models/text_pipeline/train.csv"
)

test_df = pd.read_csv(
    "models/text_pipeline/test.csv"
)

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


X_text = [
    text_to_sequence(text)
    for text in test_texts
]

X_text = np.array(X_text)


# =====================================
# CONVERT TO TENSORS
# =====================================

X_speech_tensor = torch.tensor(
    X_speech,
    dtype=torch.float32
).to(device)

X_text_tensor = torch.tensor(
    X_text,
    dtype=torch.long
).to(device)


# =====================================
# LOAD SPEECH MODEL
# =====================================

speech_model = SpeechEmotionModel(
    num_classes=7
)

speech_model.load_state_dict(
    torch.load(
        "models/speech_pipeline/speech_emotion_model.pth",
        map_location=device
    )
)

speech_model = speech_model.to(device)

speech_model.eval()

print(
    "\nSpeech model loaded successfully."
)


# =====================================
# LOAD TEXT MODEL
# =====================================

text_model = TextEmotionModel(
    vocab_size=len(vocab)
)

text_model.load_state_dict(
    torch.load(
        "models/text_pipeline/text_emotion_model.pth",
        map_location=device
    )
)

text_model = text_model.to(device)

text_model.eval()

print(
    "Text model loaded successfully."
)


# =====================================
# EXTRACT SPEECH EMBEDDINGS
# =====================================

with torch.no_grad():

    x = X_speech_tensor.permute(
        0,
        2,
        1
    )

    output, (hidden, cell) = (
        speech_model.lstm(x)
    )

    hidden_forward = hidden[-2]

    hidden_backward = hidden[-1]

    speech_embeddings = torch.cat(
        (
            hidden_forward,
            hidden_backward
        ),
        dim=1
    )

speech_embeddings = (
    speech_embeddings
    .cpu()
    .numpy()
)

print(
    "\nSpeech Embeddings Shape:",
    speech_embeddings.shape
)


# =====================================
# EXTRACT TEXT EMBEDDINGS
# =====================================

with torch.no_grad():

    embedded = text_model.embedding(
        X_text_tensor
    )

    output, (hidden, cell) = (
        text_model.lstm(embedded)
    )

    hidden_forward = hidden[-2]

    hidden_backward = hidden[-1]

    text_embeddings = torch.cat(
        (
            hidden_forward,
            hidden_backward
        ),
        dim=1
    )

text_embeddings = (
    text_embeddings
    .cpu()
    .numpy()
)

print(
    "Text Embeddings Shape:",
    text_embeddings.shape
)


# =====================================
# COMBINE FEATURES
# =====================================

fusion_features = np.concatenate(
    (
        speech_embeddings,
        text_embeddings
    ),
    axis=1
)

print(
    "\nFusion Feature Shape:",
    fusion_features.shape
)


# =====================================
# SAVE FUSION FEATURES
# =====================================

np.save(
    "models/fusion_pipeline/fusion_X.npy",
    fusion_features
)

np.save(
    "models/fusion_pipeline/fusion_y.npy",
    y_speech
)

print(
    "\nFusion preprocessing completed successfully."
)