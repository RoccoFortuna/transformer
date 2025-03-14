configs = [
    {"n_heads": 4, "num_layers": 2, "ff_hidden_dim": 512, "embed_dim": 256},  # Smallest model
    {"n_heads": 4, "num_layers": 2, "ff_hidden_dim": 1024, "embed_dim": 256},  # Larger FFN
    {"n_heads": 4, "num_layers": 3, "ff_hidden_dim": 512, "embed_dim": 256},  # More layers
    {"n_heads": 8, "num_layers": 2, "ff_hidden_dim": 512, "embed_dim": 512},  # Larger embed
    {"n_heads": 8, "num_layers": 3, "ff_hidden_dim": 512, "embed_dim": 512},  # More layers
    {"n_heads": 8, "num_layers": 3, "ff_hidden_dim": 1024, "embed_dim": 512},  # Largest
]
