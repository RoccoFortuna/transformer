import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from models.encoder import TransformerEncoder
from data.dataset import get_dataloaders
from models.config import vocab_size, max_seq_len, num_classes, batch_size, max_epochs, early_stopping_threshold, patience, learning_rate, device
from models.hparam_configs import configs  # 🔹 Import the predefined configurations

# 🔹 Create Output Directory for Plots
os.makedirs("results", exist_ok=True)

# Load dataset once
print("Loading dataset...")
train_loader, test_loader = get_dataloaders(batch_size=batch_size, max_length=max_seq_len)
print(f"✅ Dataset loaded! Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# 🔄 Run Training for Each Config
results = []

for i, config in enumerate(configs):
    print(f"\n🚀 Running Config {i+1}/{len(configs)}: {config}")

    # 🔹 Model & Classifier
    model = TransformerEncoder(
        vocab_size,
        config["embed_dim"],
        config["n_heads"],
        config["ff_hidden_dim"],
        config["num_layers"],
        max_seq_len
    ).to(device)
    classifier = nn.Linear(config["embed_dim"], num_classes).to(device)

    # 🔹 Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)

    train_losses, val_losses = [], []
    best_loss = float("inf")
    epochs_no_improve = 0  # Early stopping counter

    # 🔄 Training Loop
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0

        with tqdm(total=len(train_loader), desc=f"Training {i+1}/{len(configs)} - Epoch {epoch+1}") as pbar:
            for input_ids, labels in train_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(input_ids)
                pooled_output = outputs.mean(dim=1)  # Avg pool over sequence
                logits = classifier(pooled_output)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"✅ Epoch {epoch+1} Training Complete - Avg Loss: {avg_train_loss:.4f}")

        # 🔹 Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f"Validating {i+1}/{len(configs)} - Epoch {epoch+1}") as pbar:
                for input_ids, labels in test_loader:
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    outputs = model(input_ids)
                    pooled_output = outputs.mean(dim=1)
                    logits = classifier(pooled_output)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
                    pbar.update(1)

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        print(f"🔹 Validation Loss: {avg_val_loss:.4f}")

        # 🔹 Early Stopping
        if best_loss - avg_train_loss < early_stopping_threshold:
            epochs_no_improve += 1
            print(f"⚠️ No significant improvement: {best_loss - avg_train_loss:.5f} < {early_stopping_threshold}. Count: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"🚨 Early stopping triggered at epoch {epoch+1} for config {i+1}!")
                break
        else:
            best_loss = avg_train_loss
            epochs_no_improve = 0  # Reset

    # 🔹 Save Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", linestyle="-", label="Train Loss", color="b")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", linestyle="--", label="Val Loss", color="r")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss - {config}")
    plt.legend()
    plt.grid()

    plot_filename = f"results/loss_plot_{i+1}_heads{config['n_heads']}_layers{config['num_layers']}_ff{config['ff_hidden_dim']}_embed{config['embed_dim']}.pdf"
    plt.savefig(plot_filename)
    plt.close()

    print(f"📊 Saved plot: {plot_filename}")

print("🏆 All training runs completed!")
