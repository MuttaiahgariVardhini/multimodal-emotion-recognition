import torch
import torch.nn as nn


class SpeechEmotionModel(nn.Module):

    def __init__(self, num_classes=7):

        super(SpeechEmotionModel, self).__init__()

        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # Input shape:
        # (batch_size, 40, 150)

        # Convert to:
        # (batch_size, 150, 40)

        x = x.permute(0, 2, 1)

        # LSTM
        output, (hidden, cell) = self.lstm(x)

        # Concatenate forward + backward hidden states
        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_combined = torch.cat(
            (hidden_forward, hidden_backward),
            dim=1
        )

        # Classification
        out = self.fc(hidden_combined)

        return out