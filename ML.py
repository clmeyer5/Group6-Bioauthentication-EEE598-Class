import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib


class VAE(nn.Module):
    """Variational Autoencoder for EMG feature extraction."""
    
    def __init__(self, ts_timesteps, ts_channels, discrete_features, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Time-series encoder
        self.ts_encoder = nn.Sequential(
            nn.Conv1d(ts_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2)
        )
        
        ts_flat_size = 64 * (ts_timesteps // 4)
        
        # Discrete encoder
        self.dsc_encoder = nn.Sequential(
            nn.Linear(discrete_features, 64),
            nn.LeakyReLU()
        )
        dsc_flat_size = 64
        
        self.combined_input_size = ts_flat_size + dsc_flat_size
        
        # Latent layers
        self.fc_mu = nn.Linear(self.combined_input_size, latent_dim)
        self.fc_logvar = nn.Linear(self.combined_input_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.combined_input_size)
        
        self.ts_reverse_size = 64 * (ts_timesteps // 4)
        self.ts_decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, ts_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.dsc_decoder = nn.Sequential(
            nn.Linear(dsc_flat_size, discrete_features),
            nn.Sigmoid()
        )
    
    def encode(self, x_ts, x_dsc):
        ts_emb = self.ts_encoder(x_ts.permute(0, 2, 1))
        ts_emb = ts_emb.flatten(start_dim=1)
        
        dsc_emb = self.dsc_encoder(x_dsc)
        
        combined_emb = torch.cat((ts_emb, dsc_emb), dim=1)
        
        mu = self.fc_mu(combined_emb)
        log_var = self.fc_logvar(combined_emb)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, ts_timesteps, ts_channels):
        combined_input = self.decoder_input(z)
        
        ts_recon_size = self.ts_reverse_size
        ts_recon_input = combined_input[:, :ts_recon_size]
        dsc_recon_input = combined_input[:, ts_recon_size:]
        
        ts_recon_input = ts_recon_input.view(-1, 64, ts_timesteps // 4)
        ts_recon = self.ts_decoder(ts_recon_input).permute(0, 2, 1)
        
        dsc_recon = self.dsc_decoder(dsc_recon_input)
        
        return ts_recon, dsc_recon
    
    def forward(self, x_ts, x_dsc):
        mu, log_var = self.encode(x_ts, x_dsc)
        z = self.reparameterize(mu, log_var)
        ts_recon, dsc_recon = self.decode(z, x_ts.shape[1], x_ts.shape[2])
        return ts_recon, dsc_recon, mu, log_var
    
    def vae_loss(self, recon_ts, recon_dsc, x_ts, x_dsc, mu, log_var):
        recon_loss_ts = nn.functional.mse_loss(recon_ts, x_ts, reduction='sum')
        recon_loss_dsc = nn.functional.mse_loss(recon_dsc, x_dsc, reduction='sum')
        recon_loss = recon_loss_ts + recon_loss_dsc
        
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return (recon_loss + kl_div) / x_ts.size(0)


def train_vae(vae_model, train_loader, val_loader, device, epochs=100, lr=1e-3, patience=10):
    """Train VAE with early stopping."""
    vae_model.to(device)
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training
        vae_model.train()
        train_loss = 0.0
        for x_ts, x_dsc, _ in train_loader:
            x_ts, x_dsc = x_ts.to(device), x_dsc.to(device)
            
            optimizer.zero_grad()
            ts_recon, dsc_recon, mu, log_var = vae_model(x_ts, x_dsc)
            loss = vae_model.vae_loss(ts_recon, dsc_recon, x_ts, x_dsc, mu, log_var)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        vae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_ts, x_dsc, _ in val_loader:
                x_ts, x_dsc = x_ts.to(device), x_dsc.to(device)
                ts_recon, dsc_recon, mu, log_var = vae_model(x_ts, x_dsc)
                loss = vae_model.vae_loss(ts_recon, dsc_recon, x_ts, x_dsc, mu, log_var)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = vae_model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    vae_model.load_state_dict(best_model_state)
    return vae_model, history


class HybridClassifier:
    """VAE feature extractor + SVM classifier."""
    
    def __init__(self, vae_model, device, kernel='rbf', C=1.0):
        self.vae_model = vae_model
        self.device = device
        self.svm = SVC(kernel=kernel, C=C, class_weight='balanced', random_state=42)
        self.encoder = vae_model.encode
    
    def extract_latent_features(self, data_loader):
        """Extract latent features from VAE encoder."""
        self.vae_model.eval()
        latent_features = []
        labels = []
        
        with torch.no_grad():
            for x_ts, x_dsc, y_ohe in data_loader:
                x_ts, x_dsc = x_ts.to(self.device), x_dsc.to(self.device)
                mu, _ = self.encoder(x_ts, x_dsc)
                
                latent_features.append(mu.cpu().numpy())
                labels.append(y_ohe.cpu().numpy())
        
        Z = np.concatenate(latent_features, axis=0)
        Y_ohe = np.concatenate(labels, axis=0)
        Y_indices = np.argmax(Y_ohe, axis=1)
        
        return Z, Y_indices
    
    def train_baseline_svm(self, train_loader, val_loader):
        """Train initial SVM on training data."""
        print("Training baseline SVM...")
        
        Z_train, Y_train = self.extract_latent_features(train_loader)
        Z_val, Y_val = self.extract_latent_features(val_loader)
        
        Z_full = np.concatenate((Z_train, Z_val), axis=0)
        Y_full = np.concatenate((Y_train, Y_val), axis=0)
        
        self.svm.fit(Z_full, Y_full)
        
        val_score = self.svm.score(Z_val, Y_val)
        print(f"Baseline SVM trained. Val Accuracy: {val_score:.4f}")
        
        return val_score
    
    def adapt_to_subjects(self, enroll_loader, train_loader=None):
        """Adapt SVM using enrollment data from new subjects."""
        print("Adapting SVM to new subjects...")
        
        Z_enroll, Y_enroll = self.extract_latent_features(enroll_loader)
        
        if train_loader is not None:
            Z_train, Y_train = self.extract_latent_features(train_loader)
            Z_combined = np.concatenate((Z_train, Z_enroll), axis=0)
            Y_combined = np.concatenate((Y_train, Y_enroll), axis=0)
        else:
            Z_combined = Z_enroll
            Y_combined = Y_enroll
        
        self.svm.fit(Z_combined, Y_combined)
        
        enroll_score = self.svm.score(Z_enroll, Y_enroll)
        print(f"SVM adapted. Enrollment Accuracy: {enroll_score:.4f}")
        
        return enroll_score
    
    def evaluate(self, eval_loader, verbose=True):
        """Evaluate on test data."""
        Z_eval, Y_eval = self.extract_latent_features(eval_loader)
        Y_pred = self.svm.predict(Z_eval)
        
        accuracy = accuracy_score(Y_eval, Y_pred)
        
        if verbose:
            print(f"\nEvaluation Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(Y_eval, Y_pred, zero_division=0))
        
        return {
            'accuracy': accuracy,
            'predictions': Y_pred,
            'true_labels': Y_eval,
            'confusion_matrix': confusion_matrix(Y_eval, Y_pred)
        }
    
    def save_models(self, vae_path, svm_path):
        """Save both VAE and SVM."""
        torch.save(self.vae_model.state_dict(), vae_path)
        joblib.dump(self.svm, svm_path)
        print(f"Models saved to {vae_path} and {svm_path}")
    
    @staticmethod
    def load_models(vae_params, vae_path, svm_path, device):
        """Load saved models."""
        vae_model = VAE(**vae_params).to(device)
        vae_model.load_state_dict(torch.load(vae_path, map_location=device))
        vae_model.eval()
        
        svm_model = joblib.load(svm_path)
        
        hybrid = HybridClassifier(vae_model, device)
        hybrid.svm = svm_model
        
        print(f"Models loaded from {vae_path} and {svm_path}")
        return hybrid
