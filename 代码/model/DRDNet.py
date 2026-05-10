import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np
from torchinfo import summary
from mamba_ssm import Mamba


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = 0
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)

    def _update_cos_sin_tables(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x):
        seq_len = x.shape[2]
        self._update_cos_sin_tables(x, seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return self._apply_rotary_pos_emb(x, cos, sin)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated * sin)

class _AttentionBlock(nn.Module):
    def __init__(self, d_model, key_dim, n_head, dropout):
        super(_AttentionBlock, self).__init__()

        self.n_head = n_head

        self.w_qs = nn.Linear(d_model, n_head * key_dim)
        self.w_ks = nn.Linear(d_model, n_head * key_dim)
        self.w_vs = nn.Linear(d_model, n_head * key_dim)

        self.fc = nn.Linear(n_head * key_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.rope = RotaryPositionalEmbedding(key_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        q = rearrange(self.w_qs(x), 'b l (head k) -> head b l k', head=self.n_head)
        k = rearrange(self.w_ks(x), 'b t (head k) -> head b t k', head=self.n_head)
        v = rearrange(self.w_vs(x), 'b t (head v) -> head b t v', head=self.n_head)

        q = self.rope(q)
        k = self.rope(k)

        attn = torch.einsum('hblk, hbtk -> hblt', [q, k]) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim=3)

        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = output + residual

        return output

class FFN(nn.Module):
    def __init__(self, d_model, expansion, dropout):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.w2 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.w3 = nn.Linear(d_model * expansion, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        x = self.w3(F.silu(self.w1(x)) * self.w2(x))

        x = self.dropout(x)
        x = x + residual

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.5):
        super(TransformerEncoder, self).__init__()

        self.attention = _AttentionBlock(d_model, key_dim=8, n_head=n_head, dropout=dropout)
        self.ffn = FFN(d_model, expansion=4, dropout=dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x

class Bi_Mambablock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super(Bi_Mambablock, self).__init__()
        self.fwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, x):
        out_fwd = self.fwd_mamba(x)
        x_bwd = torch.flip(x, dims=[1])
        out_bwd = self.bwd_mamba(x_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])
        concat_features = torch.cat([out_fwd, out_bwd], dim=-1)
        g = self.gate_proj(concat_features)
        self.saved_g = g.detach().cpu().numpy()
        out = g * out_fwd + (1.0 - g) * out_bwd

        return out


class DRDNet(nn.Module):
    def __init__(self, n_classes):
        super(DRDNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (32, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()

        self.pool1 = nn.AvgPool2d((1, 45), (1, 45))
        self.pool2 = nn.MaxPool2d((1, 45), (1, 45))

        self.dropout1 = nn.Dropout(0.5)

        self.mamba = nn.ModuleList([Bi_Mambablock(d_model=16, d_state=16, d_conv=4, expand=2) for i in range(1)])
        self.transformer = nn.ModuleList([TransformerEncoder(d_model=16, n_head=10) for i in range(1)])

        self.gate_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 2)
        )

        self.lstm = nn.LSTM(input_size=16, hidden_size=64, batch_first=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # (b, 1, 32, 1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)

        x_avg = self.pool1(x)
        x_max = self.pool2(x)

        x_avg = x_avg.squeeze(dim=2).permute(0, 2, 1)  # (b, l, 16)
        x_max = x_max.squeeze(dim=2).permute(0, 2, 1)  # (b, l, 16)

        x_avg = self.dropout1(x_avg)
        x_max = self.dropout1(x_max)

        feat_mamba = x_avg
        for encoder in self.mamba:
            feat_mamba = encoder(feat_mamba)  # (b, l, 16)

        feat_transformer = x_max
        for encoder in self.transformer:
            feat_transformer = encoder(feat_transformer)  # (b, l, 16)

        combined_feats = torch.cat([feat_mamba, feat_transformer], dim=-1)

        gate_logits = self.gate_net(combined_feats)  # (b, l, 2)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (b, l, 2)

        w_mamba = gate_weights[:, :, 0:1]
        w_trans = gate_weights[:, :, 1:2]

        self.saved_alpha = w_mamba.detach().cpu().numpy()

        x_fused = w_mamba * feat_mamba + w_trans * feat_transformer

        output, (h_n, c_n) = self.lstm(x_fused)

        x = h_n.permute(1, 0, 2)  # (b, 1, 64)
        x = self.flatten(x)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    input_size = (9, 1, 32, 1000)
    model = DRDNet(6).to("cuda")
    summary(model, input_size)