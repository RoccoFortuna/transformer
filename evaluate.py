import torch
import matplotlib.pyplot as plt
import seaborn as sns
from models.encoder import TransformerEncoder
from data.dataset import get_dataloaders

# Load data
train_loader, _ = get_dataloaders(batch_size=1, max_length=50)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerEncoder(vocab_size=30522, embed_dimension=512, n_heads=8, ff_hidden_dim=1024, num_layers=3, max_seq_len=50).to(device)
model.eval()

# Get a single example
input_ids, _ = next(iter(train_loader))
input_ids = input_ids.to(device)

# Forward pass
with torch.no_grad():
    outputs = model(input_ids)

# Get attention scores from last layer
attention_scores = model.encoder_blocks[-1].attention.scaled_dot_product_attention(*model.encoder_blocks[-1].attention.forward(input_ids))

# Plot attention scores
attention_scores = attention_scores[1].squeeze().cpu().numpy()  # Shape (num_heads, seq_len, seq_len)
num_heads = attention_scores.shape[0]

fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))
for i in range(num_heads):
    sns.heatmap(attention_scores[i], ax=axes[i], cmap="viridis")
    axes[i].set_title(f"Head {i+1}")

plt.show()
