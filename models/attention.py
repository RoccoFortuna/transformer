import torch
import torch.nn.functional as F

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dimension, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = embed_dimension // n_heads
        self.Q_W = torch.nn.Linear(embed_dimension, self.d_k * n_heads)
        self.K_W = torch.nn.Linear(embed_dimension, self.d_k * n_heads)
        self.V_W = torch.nn.Linear(embed_dimension, self.d_k * n_heads)
        self.final_linear = torch.nn.Linear(self.d_k * n_heads, embed_dimension)


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V), attention_weights

    def forward(self, input):
        # input.size = (batch_size, seq_len, embed_dimension)
        batch_size, seq_len, embed_dimension = input.size()

        Q = self.Q_W(input)  # (batch_size, seq_len, embed_dim)
        K = self.K_W(input)
        V = self.V_W(input)

        # split into multiple heads
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)


        scaled_dot_product_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        output = scaled_dot_product_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len, n_heads, d_k)
        output = output.reshape(batch_size, seq_len, -1)
        return self.final_linear(output)
