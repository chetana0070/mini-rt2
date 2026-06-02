import torch
import torch.nn as nn


class MiniRT2TokenPolicy(nn.Module):
    """
    Mini RT-2-style policy.

    Input:
        image: [B, 3, H, W]
        text_ids: [B, L]

    Output:
        logits: [B, action_dim, num_bins]

    Meaning:
        image + instruction -> action tokens
    """

    def __init__(
        self,
        vocab_size=128,
        num_bins=256,
        action_dim=7,
        text_dim=64,
        image_dim=128,
        hidden_dim=256,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.action_dim = action_dim

        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, image_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.text_embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)

        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim * num_bins),
        )

    def encode_text(self, text_ids):
        emb = self.text_embed(text_ids)

        mask = (text_ids != 0).float().unsqueeze(-1)
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)

        return summed / denom

    def forward(self, image, text_ids):
        img_feat = self.vision(image)
        txt_feat = self.encode_text(text_ids)

        fused = torch.cat([img_feat, txt_feat], dim=-1)

        logits = self.fusion(fused)
        logits = logits.view(-1, self.action_dim, self.num_bins)

        return logits

    def predict_tokens(self, image, text_ids):
        logits = self.forward(image, text_ids)
        return logits.argmax(dim=-1)
