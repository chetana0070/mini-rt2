import torch
import torch.nn as nn


class MiniRT2TokenPolicyV2(nn.Module):
    """
    Mini RT-2 v2.

    Input:
        image
        text instruction
        proprio: eef_pos + cube_pos + relative vector + timestep

    Output:
        7 action tokens
    """

    def __init__(
        self,
        vocab_size=128,
        num_bins=256,
        action_dim=7,
        text_dim=64,
        image_dim=128,
        proprio_dim=10,
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

        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim + 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
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

    def forward(self, image, text_ids, proprio):
        img_feat = self.vision(image)
        txt_feat = self.encode_text(text_ids)
        prop_feat = self.proprio_mlp(proprio)

        fused = torch.cat([img_feat, txt_feat, prop_feat], dim=-1)

        logits = self.fusion(fused)
        logits = logits.view(-1, self.action_dim, self.num_bins)

        return logits

    def predict_tokens(self, image, text_ids, proprio):
        logits = self.forward(image, text_ids, proprio)
        return logits.argmax(dim=-1)
