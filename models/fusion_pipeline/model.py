import torch
import torch.nn as nn


class FusionEmotionModel(nn.Module):

    def __init__(self):

        super(FusionEmotionModel, self).__init__()

        # Speech feature size
        speech_input_size = 256

        # Text feature size
        text_input_size = 256

        # Combined feature size
        fusion_input_size = (
            speech_input_size +
            text_input_size
        )

        # Fully connected layers

        self.fc1 = nn.Linear(
            fusion_input_size,
            128
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(
            128,
            7
        )

    def forward(
        self,
        speech_features,
        text_features
    ):

        # Concatenate features

        fused = torch.cat(
            (
                speech_features,
                text_features
            ),
            dim=1
        )

        x = self.fc1(fused)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x