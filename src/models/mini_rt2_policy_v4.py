import torch
import torch.nn as nn


class MiniRT2StagePolicyV4(nn.Module):
    """
    Stage-conditioned Mini RT-2.

    Inputs:
        image
        text
        proprio
        stage_id

    Outputs:
        action token logits: [B, 7, 256]
        predicted stage logits: [B, 5]
    """

    def __init__(
        self,
        vocab_size=128,
        num_bins=256,
        action_dim=7,
        num_stages=5,
        text_dim=64,
        image_dim=128,
        proprio_dim=10,
        stage_dim=32,
        hidden_dim=384,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.action_dim = action_dim
        self.num_stages = num_stages

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

        self.stage_embed = nn.Embedding(num_stages, stage_dim)

        base_dim = image_dim + text_dim + 64

        self.stage_head = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_stages),
        )

        self.action_head = nn.Sequential(
            nn.Linear(base_dim + stage_dim, hidden_dim),
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

    def encode_base(self, image, text_ids, proprio):
        img_feat = self.vision(image)
        txt_feat = self.encode_text(text_ids)
        prop_feat = self.proprio_mlp(proprio)

        return torch.cat([img_feat, txt_feat, prop_feat], dim=-1)

    def forward(self, image, text_ids, proprio, stage_id):
        base = self.encode_base(image, text_ids, proprio)

        stage_logits = self.stage_head(base)

        stage_feat = self.stage_embed(stage_id)
        fused = torch.cat([base, stage_feat], dim=-1)

        action_logits = self.action_head(fused)
        action_logits = action_logits.view(-1, self.action_dim, self.num_bins)

        return action_logits, stage_logits

    def predict(self, image, text_ids, proprio, stage_id):
        action_logits, stage_logits = self.forward(image, text_ids, proprio, stage_id)
        action_tokens = action_logits.argmax(dim=-1)
        pred_stage = stage_logits.argmax(dim=-1)
        return action_tokens, pred_stage
