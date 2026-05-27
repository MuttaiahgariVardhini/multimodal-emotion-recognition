import os
import librosa
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# =====================================
# DATASET PATH
# =====================================

DATASET_PATH = "data/TESS Toronto emotional speech set data"


# =====================================
# STORE FILE PATHS + EMOTIONS
# =====================================

file_paths = []
emotions = []


# =====================================
# READ DATASET FOLDERS
# =====================================

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    # Skip non-folder files
    if not os.path.isdir(folder_path):
        continue

    # Read wav files
    for file_name in os.listdir(folder_path):

        if file_name.endswith(".wav"):

            # Full audio path
            full_path = os.path.join(folder_path, file_name)

            # Extract emotion
            emotion = folder.split("_")[-1].lower()

            # Fix inconsistent labels
            if emotion == "surprised":
                emotion = "surprise"

            file_paths.append(full_path)
            emotions.append(emotion)


# =====================================
# CREATE DATAFRAME
# =====================================

df = pd.DataFrame({
    "path": file_paths,
    "emotion": emotions
})


print(df.head())

print("\nTotal Samples:", len(df))

print("\nEmotion Distribution:")
print(df["emotion"].value_counts())


# =====================================
# MFCC FEATURE EXTRACTION
# =====================================

def extract_mfcc(file_path, max_pad_length=150):

    try:

        # Load audio
        audio, sample_rate = librosa.load(
            file_path,
            sr=16000
        )

        # Remove silence
        audio, _ = librosa.effects.trim(audio)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Pad shorter sequences
        if mfcc.shape[1] < max_pad_length:

            pad_width = max_pad_length - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant'
            )

        # Truncate longer sequences
        else:

            mfcc = mfcc[:, :max_pad_length]

        return mfcc

    except Exception as e:

        print("Error processing:", file_path)
        print(e)

        return None


# =====================================
# TEST MFCC EXTRACTION
# =====================================

sample_path = df.iloc[0]["path"]

mfcc_features = extract_mfcc(sample_path)

print("\nMFCC Shape:", mfcc_features.shape)


# =====================================
# EXTRACT FEATURES FROM ALL FILES
# =====================================

X = []
y = []

print("\nExtracting MFCC features from all audio files...\n")

for index, row in df.iterrows():

    features = extract_mfcc(row["path"])

    if features is not None:

        X.append(features)
        y.append(row["emotion"])


# =====================================
# CONVERT TO NUMPY ARRAYS
# =====================================

X = np.array(X)
y = np.array(y)

print("Feature Shape:", X.shape)
print("Labels Shape:", y.shape)


# =====================================
# NORMALIZE FEATURES
# =====================================

X = (X - np.mean(X)) / np.std(X)

print("\nFeatures normalized successfully.")


# =====================================
# LABEL ENCODING
# =====================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEmotion Classes:")
print(label_encoder.classes_)


# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# =====================================
# SAVE PREPROCESSED DATA
# =====================================

np.save("models/speech_pipeline/X_train.npy", X_train)
np.save("models/speech_pipeline/X_test.npy", X_test)

np.save("models/speech_pipeline/y_train.npy", y_train)
np.save("models/speech_pipeline/y_test.npy", y_test)

print("\nPreprocessed data saved successfully.")