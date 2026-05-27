# Multimodal Emotion Recognition using Speech, Text, and Fusion Models

## Project Overview

This project implements a Multimodal Emotion Recognition System using:

- Speech Emotion Recognition
- Text Emotion Recognition
- Fusion-based Emotion Recognition

The system predicts the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Dataset Used:
- Toronto Emotional Speech Set (TESS)

---

# Dataset
Dataset link: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

The TESS dataset contains:

- Audio `.wav` files
- Emotion labels
- Spoken words with emotional variations

Total Samples:
- 2800 audio samples

Emotion Classes:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

# Models Used

## Speech Pipeline
- MFCC Feature Extraction
- BiLSTM Model

## Text Pipeline
- Embedding + BiLSTM Model

## Fusion Pipeline
- Fusion of Speech and Text Embeddings
- Fully Connected Neural Network

---

# Folder Structure

project/
├── data/
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── Results/
│   └── plots/
│
├── README.md
├── requirements.txt
-->Report

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Librosa

---

# Installation

## Clone Repository
Optional(if needed)
Please download it from the provided link and place it inside:

data/TESS Toronto emotional speech set data/

git clone https://github.com/MuttaiahgaruVardhini/multimodal-emotion-recognition
cd multimodal-emotion-recognition

## Install Requirements

pip install -r requirements.txt

---

# Run Instructions

## Speech Pipeline

python models/speech_pipeline/preprocess.py

python models/speech_pipeline/train.py

python models/speech_pipeline/test.py

---

## Text Pipeline

python models/text_pipeline/preprocess.py

python models/text_pipeline/train.py

python models/text_pipeline/test.py

---

## Fusion Pipeline

python models/fusion_pipeline/preprocess.py

python models/fusion_pipeline/train.py

python models/fusion_pipeline/test.py

---

# Results

| Model | Test Accuracy |
|-------|----------------|
| Speech Model | 97% – 98% |
| Text Model | ~1% |
| Fusion Model | 97%  |

---

# Visualizations

The project generates:

- Confusion Matrix
- Classification Report
- t-SNE Visualization

Saved inside:

Results/plots/

---

# Observations

- Speech emotion recognition achieved high accuracy.
- Text-only emotion recognition performed poorly because the dataset contains isolated words instead of full sentences.
- Fusion learning improved robustness and overall performance.

---

# Future Improvements

- Use BERT for text emotion recognition
- Use Transformer-based speech models
- Add real-time emotion prediction
- Build web application interface

---

# Conclusion

This project successfully compares:
- Speech Emotion Recognition
- Text Emotion Recognition
- Multimodal Fusion Emotion Recognition

The results show that multimodal fusion improves overall emotion recognition performance.