# markov_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MarkovTransitionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MarkovTransitionModule, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x_prev = x[:, :-1, :]
        c_prev = F.gelu(x_prev)  # simple local context
        t_logits = self.W1(x_prev) + self.W2(c_prev) + self.bias
        t_probs = F.softmax(t_logits, dim=-1)
        padded = F.pad(t_probs, (0, 0, 1, 0))  # pad to match input size
        return padded


class SelfAttentionModule(nn.Module):
    def __init__(self, dim, n_heads):
        super(SelfAttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return attn_output


class FusionModule(nn.Module):
    def __init__(self, dim):
        super(FusionModule, self).__init__()
        self.gate = nn.Linear(2 * dim, 1)

    def forward(self, h_mtm, h_sam):
        concat = torch.cat([h_mtm, h_sam], dim=-1)
        alpha = torch.sigmoid(self.gate(concat))
        fused = alpha * h_mtm + (1 - alpha) * h_sam
        return fused


class MarkovianTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super(MarkovianTransformer, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.mtm = MarkovTransitionModule(hidden_dim, hidden_dim)
        self.sam = SelfAttentionModule(hidden_dim, n_heads)
        self.fusion = FusionModule(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)  # For binary classification

    def forward(self, x):
        x = self.encoder(x)
        h_mtm = self.mtm(x)
        h_sam = self.sam(x)
        h_fused = self.fusion(h_mtm, h_sam)
        logits = self.classifier(h_fused[:, -1, :])  # Use final token for prediction
        return logits


if __name__ == "__main__":
    # Example usage
    model = MarkovianTransformer(input_dim=16, hidden_dim=64, n_heads=4)
    dummy_input = torch.randn(8, 50, 16)  # (batch, seq_len, features)
    output = model(dummy_input)
    print("Logits:", output.shape)
