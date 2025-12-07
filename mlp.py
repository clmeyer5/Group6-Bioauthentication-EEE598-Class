import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=35, dropout_rate=0.4, device='cpu'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.device = device

        # 1. Feature Extractor
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate)
        )

        # 2. The Classifier Head 
        self.head = nn.Linear(hidden_dim // 2, num_classes)

        self.to(self.device)

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        features = self.body(x)
        return self.head(features)

    def save_model(self, file_path):
        current_dev = self.device
        self.to('cpu')
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        self.to(current_dev)

    def load_model(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path, map_location=self.device))
            self.to(self.device)
            print("Model loaded successfully.")
        else:
            print(f"Warning: No model found at {file_path}")

    # ==========================================
    # Phase 1: Pre-training
    # ==========================================
    def train_pretraining(self, train_loader, val_loader=None, model_name="mlp_pretrain", lr=0.001, epochs=50):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        print(f"\n--- Starting Pre-training on {self.device} ---")

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).long()

                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            avg_loss = total_loss / len(train_loader)
            acc = 100 * correct / total

            if val_loader:
                scheduler.step(avg_loss)

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        self.save_model(f"{model_name}.pth")

    # ==========================================
    # Phase 2: Transfer Learning Setup 
    # ==========================================
    def prepare_for_finetuning(self, fine_tune_hidden=64, unfreeze_body=True):
        """
        Args:
            fine_tune_hidden: Size of the new intermediate layer in the head.
            unfreeze_body: If True, allows the body layers to update (at a lower LR).
        """
        print("\n--- Modifying Architecture for Binary Verification ---")

        # 1. Determine Input Features
        self.eval()

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_dim).to(self.device)
            dummy_output = self.body(dummy_input)
            in_features = dummy_output.shape[1]

        self.train()

        # 2. Handle Body Freezing/Unfreezing
        if unfreeze_body:
            print(">> Strategy: Unfreezing Body (Discriminative Fine-Tuning)")
            for param in self.body.parameters():
                param.requires_grad = True
        else:
            print(">> Strategy: Freezing Body (Feature Extraction Only)")
            for param in self.body.parameters():
                param.requires_grad = False

        # 3. Create a Deeper, More Robust Head
        self.head = nn.Sequential(
            nn.Linear(in_features, fine_tune_hidden),
            nn.BatchNorm1d(fine_tune_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5), 

            nn.Linear(fine_tune_hidden, fine_tune_hidden // 2),
            nn.BatchNorm1d(fine_tune_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(fine_tune_hidden // 2, 1)
        )

        self.to(self.device)

    # ==========================================
    # Phase 3: Fine-tuning 
    # ==========================================
    def train_finetuning(self, train_loader, pos_weight=1.0, model_name="mlp_finetuned", head_lr=0.001, body_lr=1e-5, epochs=30):
        """
        Uses different learning rates for the Head (Fast) and Body (Slow).
        """

        pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Discriminative Learning Rates
        optimizer = optim.Adam([
            {'params': self.body.parameters(), 'lr': body_lr},
            {'params': self.head.parameters(), 'lr': head_lr}  
        ])

        print(f"\n--- Starting Fine-tuning on {self.device} ---")
        print(f"Head LR: {head_lr} | Body LR: {body_lr}")

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            # Helper to track confidence distribution
            all_preds = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)

                optimizer.zero_grad()
                logits = self(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Track raw probabilities for debugging
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    all_preds.append(probs.cpu().numpy())

            # Debugging confidence
            all_preds = np.concatenate(all_preds)
            avg_conf = np.mean(all_preds)

            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Avg Pred Conf: {avg_conf:.4f}")

        self.save_model(f"{model_name}.pth")

    def predict_score(self, x):
        self.eval()
        with torch.no_grad():
            if x.device != self.device:
                x = x.to(self.device)
            logits = self(x)
            return torch.sigmoid(logits)