import torch
import torch.nn.functional as F

from .encoder_block import TransformerEncoderBlock

class TransformerEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dimension, n_heads, ff_hidden_dim, num_layers, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dimension)

        # Positional Encoding (learnt)
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, max_seq_len, embed_dimension))

        self.encoder_blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(embed_dimension, n_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.token_embedding(x)  # (batch_size, seq_len, embed_dimension)

        x = x + self.position_embedding[:, :seq_len, :]

        x = self.dropout(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        return x










# Define model parameters
vocab_size = 10000  # Vocabulary size (e.g., a small tokenizer)
embed_dimension = 512  # Embedding size
n_heads = 8  # Number of attention heads
ff_hidden_dim = 2048  # Feedforward hidden layer size
num_layers = 6  # Number of Transformer blocks
max_seq_len = 50  # Max sequence length
batch_size = 2  # Number of sentences in a batch

# Create Transformer Encoder model
encoder = TransformerEncoder(vocab_size, embed_dimension, n_heads, ff_hidden_dim, num_layers, max_seq_len)

# Create a batch of random tokenized inputs (integers representing words)
dummy_input = torch.randint(0, vocab_size, (batch_size, max_seq_len))  # (batch_size, seq_len)

# Forward pass through the Transformer Encoder
output_embeddings = encoder(dummy_input)

# Check output shape
print("Input shape:", dummy_input.shape)  # Expected: (batch_size, seq_len)
print("Output shape:", output_embeddings.shape)  # Expected: (batch_size, seq_len, embed_dimension)

# Inspect some embeddings
print("Sample output embeddings:", output_embeddings[0, :5, :10])  # First 5 tokens, first 10 embedding dims
