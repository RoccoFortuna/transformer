import torch

# Model Configuration
vocab_size = 30522  # BERT's vocab size
embed_dim = 512
num_heads = 8
ff_hidden_dim = 1024
num_layers = 3
max_seq_len = 50
num_classes = 4  # AG News has 4 classes

# Training Configuration
batch_size = 32
max_epochs = 100  # Upper limit, but early stopping may stop earlier
early_stopping_threshold = 0.02  # Stop if loss improvement < threshold
patience = 3  # Number of epochs to wait before stopping if no improvement
learning_rate = 3e-4

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
