import os
import torch
import torch.nn as nn
import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import glob # Import glob for checking file existence

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        # Linear Projection (Tokenizer)
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model)) # Max sequence 5000

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output layer (Predicting next log return)
        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        x = self.input_projection(x) # [batch, seq_len, d_model]
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :]) # Take only the last time step for prediction
        return x

# ==========================================
# 2. DATASET LOADER (PARQUET TO TOKENS)
# ==========================================

class KassandraDataset(Dataset):
    def __init__(self, base_dir, asset_registry, seq_length=64):
        self.seq_length = seq_length
        self.all_sequences = []

        print(f"[INFO] Scanning directory: {base_dir}")

        for key, meta in asset_registry.items():
            # Construct the path to the asset's root directory (before timeframe)
            asset_root_path = os.path.join(
                base_dir,
                f"asset_class={meta['asset_class']}",
                f"symbol={meta['symbol']}"
            )

            # Check if this root directory exists
            if not os.path.exists(asset_root_path):
                print(f"[WARN] Asset root path not found for {key}: {asset_root_path}")
                continue

            # Now, construct the glob pattern to match all parquet files within all timeframe subdirectories
            parquet_glob_pattern = os.path.join(asset_root_path, "timeframe=*", "*.parquet")

            # Check if any parquet files exist before calling scan_parquet
            if not glob.glob(parquet_glob_pattern):
                print(f"[WARN] No parquet files found for {key} at {parquet_glob_pattern}")
                continue

            # Scan all parquet files matching the pattern
            df = pl.scan_parquet(parquet_glob_pattern).collect()

            if df.is_empty():
                print(f"[WARN] Collected DataFrame for {key} is empty after scan_parquet.")
                continue

            # Hitung Log Return
            # Asumsi kolom harga adalah 'Close'
            df = df.with_columns([
                (pl.col("Close").log() - pl.col("Close").shift(1).log()).alias("log_return")
            ]).drop_nulls()

            # Normalisasi sederhana (Z-Score) untuk membantu konvergensi Transformer
            log_ret = df["log_return"].to_numpy().astype(np.float32)
            mean, std = log_ret.mean(), log_ret.std() + 1e-9
            log_ret = (log_ret - mean) / std

            # Sliding Window Tokenization
            for i in range(len(log_ret) - seq_length):
                window = log_ret[i : i + seq_length]
                target = log_ret[i + seq_length]
                self.all_sequences.append((window, target))

        print(f"[OK] Total tokens/sequences created: {len(self.all_sequences)}")

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        x, y = self.all_sequences[idx]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)

# ==========================================
# 3. TRAINING / FITTING LOOP
# ==========================================

def fit_transformer(base_dir, asset_registry, epochs=5, batch_size=32):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Initialize Dataset & DataLoader
    dataset = KassandraDataset(base_dir, asset_registry, seq_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model Hyperparameters
    model = TimeSeriesTransformer(input_dim=1, d_model=64, nhead=4, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    return model

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Gunakan path Google Drive / Colab Anda
    DATA_BASE_DIR = '/content/drive/MyDrive/books/CASSANDRA/data_base'

    # Placeholder ASSET_REGISTRY dari prompt Anda
    # (Pastikan data sudah di-download ke dir tersebut menggunakan script download sebelumnya)

    # model = fit_transformer(DATA_BASE_DIR, ASSET_REGISTRY)
    pass
