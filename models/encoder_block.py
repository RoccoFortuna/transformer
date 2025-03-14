import torch
import torch.nn.functional as F
from .attention import MultiHeadAttention

class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embed_dimension, n_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dimension, n_heads)
        self.norm1 = torch.nn.LayerNorm(embed_dimension)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_dimension, ff_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_dim, embed_dimension)
        )
        self.norm2 = torch.nn.LayerNorm(embed_dimension)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        norm1_output = self.norm1(x + self.dropout(attn_output)) # batch-norm on skip connected layer

        ffn_output = self.ffn(norm1_output)
        norm2_output = self.norm2(norm1_output + self.dropout(ffn_output))
        return norm2_output
