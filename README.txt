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

Dataset Link:
https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

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

Note:
The dataset is not included in this repository due to large file size.

Please download the dataset from the provided link and place it inside:

data/TESS Toronto emotional speech set data/

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
│   │   ├── preprocess.py
│   │   ├── speech_model.py
│   │   ├── train.py
│   │   └── test.py
│   │
│   ├── text_pipeline/
│   │   ├── preprocess.py
│   │   ├── text_model.py
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── fusion_pipeline/
│       ├── preprocess.py
│       ├── model.py
│       ├── train.py
│       └── test.py
│
├── Results/
│   └── plots/
│
├── README.md
├── requirements.txt
├── Final_Report.pdf

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

```bash
git clone https://github.com/MuttaiahgaruVardhini/multimodal-emotion-recognition.git

cd multimodal-emotion-recognition
```

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# Run Instructions

## Speech Pipeline

### Preprocessing

```bash
python models/speech_pipeline/preprocess.py
```

### Training

```bash
python models/speech_pipeline/train.py
```

### Testing

```bash
python models/speech_pipeline/test.py
```

---

## Text Pipeline

### Preprocessing

```bash
python models/text_pipeline/preprocess.py
```

### Training

```bash
python models/text_pipeline/train.py
```

### Testing

```bash
python models/text_pipeline/test.py
```

---

## Fusion Pipeline

### Preprocessing

```bash
python models/fusion_pipeline/preprocess.py
```

### Training

```bash
python models/fusion_pipeline/train.py
```

### Testing

```bash
python models/fusion_pipeline/test.py
```

---

# Results

| Model | Test Accuracy |
|-------|----------------|
| Speech Model | 97% – 98% |
| Text Model | ~1% |
| Fusion Model | 97% – 98% |

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
