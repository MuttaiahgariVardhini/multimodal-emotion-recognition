import torch
import torch.nn as nn


class TextEmotionModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=128,
        num_classes=7
    ):

        super(TextEmotionModel, self).__init__()

        # =====================================
        # EMBEDDING LAYER
        # =====================================

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # =====================================
        # BiLSTM LAYER
        # =====================================

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # =====================================
        # CLASSIFIER
        # =====================================

        self.fc = nn.Sequential(

            nn.Linear(hidden_size * 2, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # =====================================
        # EMBEDDINGS
        # =====================================

        x = self.embedding(x)

        # =====================================
        # BiLSTM
        # =====================================

        output, (hidden, cell) = self.lstm(x)

        # Forward + Backward hidden states
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_combined = torch.cat(
            (hidden_forward, hidden_backward),
            dim=1
        )

        # =====================================
        # CLASSIFICATION
        # =====================================

        out = self.fc(hidden_combined)

        return out