import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

class AE(nn.Module):
    def __init__(self, input_len, input_channels=1, latent_dim=2, device='cpu'):
        super(AE, self).__init__()

        self.input_len = input_len
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.device = device

        # ==================================================
        # 1. ENCODER CNN
        # ==================================================
        self.encoder_cnn = nn.Sequential(
            # Layer 1: Input -> 32
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32 -> 64
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

       
        self.encoder_cnn.to(self.device)

        
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_len).to(self.device)
            dummy_output = self.encoder_cnn(dummy_input)
            self.flatten_dim = dummy_output.numel() 
            self.cnn_out_shape = dummy_output.shape 

        # ==================================================
        # 2. ENCODER DENSE FUNNEL
        # ==================================================
        self.encoder_dense = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, latent_dim) 
        )

        # ==================================================
        # 3. DECODER DENSE FUNNEL
        # ==================================================
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, self.flatten_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # ==================================================
        # 4. DECODER CNN
        # ==================================================
        self.decoder_cnn = nn.Sequential(
            # Inverse Layer 3: 128 -> 64
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Inverse Layer 2: 64 -> 32
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Inverse Layer 1: 32 -> Input Channels
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.to(self.device)

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        # --- ENCODE ---
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)      
        z = self.encoder_dense(x)      

        # --- DECODE ---
        x = self.decoder_dense(z)     
        x = x.view(x.size(0), self.cnn_out_shape[1], self.cnn_out_shape[2]) 
        reconstructed = self.decoder_cnn(x)

        # Safety Crop (for odd/even length mismatch)
        if reconstructed.shape[2] != self.input_len:
            reconstructed = reconstructed[:, :, :self.input_len]

        return reconstructed

    def get_latent_features(self, x):
        with torch.no_grad():
            self.eval()
            if x.device != self.device:
                x = x.to(self.device)
            x = self.encoder_cnn(x)
            x = x.view(x.size(0), -1)
            z = self.encoder_dense(x)
            return z

    def save_model(self, file_path):
        self.to('cpu')
        checkpoint = {
            'input_len': self.input_len,
            'input_channels': self.input_channels,
            'latent_dim': self.latent_dim,
            'state_dict': self.state_dict()
        }
        torch.save(checkpoint, file_path)
        print(f"Model saved to {file_path}")
        self.to(self.device)

    @classmethod
    def load_model(cls, file_path, device=None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No model found at {file_path}")

        if device is None:
            if torch.cuda.is_available(): device = 'cuda'
            elif torch.backends.mps.is_available(): device = 'mps'
            else: device = 'cpu'

        checkpoint = torch.load(file_path, map_location=device)
        model = cls(
            input_len=checkpoint['input_len'],
            input_channels=checkpoint['input_channels'],
            latent_dim=checkpoint['latent_dim'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

    def train_ae(self, train_data, model_name, LR=0.001, EPOCHS=100, batch_size=64):
        criterion = nn.MSELoss()

        # Reduce LR slightly if loss plateaus
        optimizer = optim.Adam(self.parameters(), lr=LR)

        # Add Scheduler to lower LR if loss gets stuck (Helps break out of 1.0 loss)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        dataset = torch.utils.data.TensorDataset(train_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"Starting training on {self.device}...")
        self.train()

        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in dataloader:
                x_batch = batch[0].to(self.device)

                optimizer.zero_grad()
                output = self(x_batch)

                loss = criterion(output, x_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Step the scheduler
            scheduler.step(avg_loss)

            if (epoch+1) % 5 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.5f}')

        self.save_model(f"{model_name}.pth")



def extract_ae_features(signals_batch, ae_model,scaler=None):
    """Extract AE latent features from raw cleaned signals."""
    n_trials, length, channels = signals_batch.shape

    signals_flat = signals_batch.reshape(-1, channels)
    if scaler == None:
        scaler = StandardScaler()
    signals_scaled_flat = scaler.fit_transform(signals_flat)
    signals_scaled = signals_scaled_flat.reshape(n_trials, length, channels)

    # Transpose to (batch, channels, length)
    signals_scaled = np.transpose(signals_scaled, (0, 2, 1))

    # Reshape to single-channel format (batch*channels, 1, length)
    signals_scaled = signals_scaled.reshape(-1, 1, length)

    # Convert to tensor and extract features
    signals_tensor = torch.tensor(signals_scaled, dtype=torch.float32)

    with torch.no_grad():
        latent = ae_model.get_latent_features(signals_tensor)

    # Reshape back: (batch*channels, latent_dim) -> (batch, channels*latent_dim)
    latent_np = latent.cpu().numpy()
    latent_per_trial = latent_np.reshape(n_trials, -1)

    return latent_per_trial

def get_ae_features_batched(signals, ae_model, ts_scaler, batch_size=256, device='cuda'):
    """
    Extracts features in batches.
    """
    n_trials, length, channels = signals.shape

    # Scale Data
    signals_flat = signals.reshape(-1, channels)
    signals_scaled_flat = ts_scaler.transform(signals_flat) 
    signals_scaled = signals_scaled_flat.reshape(n_trials, length, channels)

    # Transpose to (N, Ch, L) and then Reshape to (N*Ch, 1, L)
    signals_scaled = np.transpose(signals_scaled, (0, 2, 1))
    signals_scaled = signals_scaled.reshape(-1, 1, length)

    # Create DataLoader for inference
    dataset = TensorDataset(torch.tensor(signals_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ae_model.eval()
    all_latents = []

    print(f"Extracting AE features in batches...")
    with torch.no_grad():
        for batch in loader:
            x_batch = batch[0].to(device)
            # Get Latent (Batch_Size, Latent_Dim)
            z = ae_model.get_latent_features(x_batch)
            all_latents.append(z.cpu().numpy())

    # Concatenate all batches
    latent_all = np.concatenate(all_latents, axis=0)

    # Reshape back to (N_Trials, Channels * Latent_Dim)
    latent_per_trial = latent_all.reshape(n_trials, -1)
    return latent_per_trial



def inverse_transform_output(tensor_output, scaler, original_num_channels=28):
    """
    Takes (Batch*Channels, 1, Length) Tensor
    Returns (Batch, Length, Channels) Numpy Array in Original Units
    """
    # 1. Move to CPU and Numpy
    data_np = tensor_output.cpu().numpy()

    # 2. Reshape to separate Batch and Channels
    num_samples = data_np.shape[0] // original_num_channels
    length = data_np.shape[2]

    data_np = data_np.reshape(num_samples, original_num_channels, length)

    # 3. Transpose back: (Batch, Channels, Length) -> (Batch, Length, Channels)
    data_np = np.transpose(data_np, (0, 2, 1))

    # 4. Flatten for Scaler
    flat_data = data_np.reshape(-1, original_num_channels)

    # 5. Inverse Scale
    original_scale_flat = scaler.inverse_transform(flat_data)

    # 6. Reshape to original 3D format
    return original_scale_flat.reshape(num_samples, length, original_num_channels)