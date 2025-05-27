import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error

# Set random seed for reproducibility
SEED = 43
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Variational Autoencoder (VAE) Model
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        """Variational Autoencoder for feature learning and anomaly detection.
        
        Args:
            input_dim (int): Dimensionality of input data
            latent_dim (int): Dimensionality of latent space
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)     # Mean layer
        self.fc_logvar = nn.Linear(128, latent_dim) # Log variance layer

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Anomaly Detection Class
class AnomalyDetector:
    def __init__(self, 
                 train_obs: np.ndarray, 
                 val_obs: np.ndarray, 
                 test_obs: np.ndarray,
                 train_forecast: np.ndarray, 
                 val_forecast: np.ndarray, 
                 test_forecast: np.ndarray,
                 window_length: int = 10, 
                 batch_size: int = 128, 
                 root_cause: bool = False):
        """Initialize anomaly detection framework.
        
        Args:
            train_obs (np.ndarray): Training observed values
            val_obs (np.ndarray): Validation observed values
            test_obs (np.ndarray): Test observed values
            train_forecast (np.ndarray): Training predictions
            val_forecast (np.ndarray): Validation predictions
            test_forecast (np.ndarray): Test predictions
            window_length (int): Sliding window size for normalization
            batch_size (int): Training batch size
            root_cause (bool): Enable root cause analysis
        """
        self.train_obs = train_obs
        self.val_obs = val_obs
        self.test_obs = test_obs
        self.train_forecast = train_forecast
        self.val_forecast = val_forecast
        self.test_forecast = test_forecast
        self.root_cause = root_cause
        
        # Set window length
        if window_length is None:
            self.window_length = len(train_obs) + len(val_obs)
        else:
            self.window_length = window_length
            
        self.batch_size = batch_size

        if self.root_cause:
            self.val_re_full = None
            self.test_re_full = None

    def train_vae(self, 
                  train_error: np.ndarray, 
                  val_error: np.ndarray, 
                  test_error: np.ndarray,
                  latent_dim: int = 10, 
                  epochs: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Train VAE model and compute reconstruction errors.
        
        Args:
            train_error (np.ndarray): Training errors
            val_error (np.ndarray): Validation errors
            test_error (np.ndarray): Test errors
            latent_dim (int): Latent space dimension
            epochs (int): Number of training epochs
            
        Returns:
            val_re (np.ndarray): Validation reconstruction scores
            test_re (np.ndarray): Test reconstruction scores
            val_re_full (np.ndarray): Full validation reconstruction errors
            test_re_full (np.ndarray): Full test reconstruction errors
        """
        input_dim = train_error.shape[1]
        vae = VAE(input_dim, latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)

        # Convert to tensor
        train_error_tensor = torch.tensor(train_error, dtype=torch.float32).to(vae.device)

        # Training loop with progress bar
        pbar = tqdm(range(epochs), desc="Training VAE", unit="epoch")
        for epoch in pbar:
            recon_batch, mu, logvar = vae(train_error_tensor)
            mse_loss = criterion(recon_batch, train_error_tensor)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = mse_loss + kl_divergence

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pbar.set_postfix({'Total Loss': total_loss.item() / len(train_error)})

        # Evaluate on validation and test data
        with torch.no_grad():
            val_error_tensor = torch.tensor(val_error, dtype=torch.float32).to(vae.device)
            test_error_tensor = torch.tensor(test_error, dtype=torch.float32).to(vae.device)
            
            # Get reconstructions
            val_recon, _, _ = vae(val_error_tensor)
            test_recon, _, _ = vae(test_error_tensor)
            
            # Calculate reconstruction errors
            val_re_full = np.abs(val_recon.cpu().numpy() - val_error)
            test_re_full = np.abs(test_recon.cpu().numpy() - test_error)
            val_re = val_re_full.sum(axis=1)
            test_re = test_re_full.sum(axis=1)

        return val_re, test_re, val_re_full, test_re_full

    def calculate_scores(self, normalization_method: str = 'iqr') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate anomaly scores and predictions.
        
        Args:
            normalization_method (str): 'iqr' or 'z-score' for error normalization
            
        Returns:
            realtime_indicator (np.ndarray): Anomaly scores over time
            anomaly_prediction (np.ndarray): Binary anomaly predictions
            variable_contributions (np.ndarray): Variable-wise contribution to anomalies
        """
        # Compute absolute errors
        train_abs = np.abs(self.train_obs - self.train_forecast)
        val_abs = np.abs(self.val_obs - self.val_forecast)
        test_abs = np.abs(self.test_obs - self.test_forecast)

        # Calculate evaluation metrics for each variable
        for a in range(16):
            r2 = r2_score(self.test_obs[:-301, a], self.test_forecast[:-301, a])
            mae = mean_absolute_error(self.test_obs[:-301, a], self.test_forecast[:-301, a])
            print(f"Variable {a}: R²={r2:.4f}, MAE={mae:.4f}")

        # Normalize errors
        train_norm = self.normalize_errors(train_abs, method=normalization_method)
        val_norm = self.normalize_errors(val_abs, method=normalization_method)
        
        # Train VAE and get reconstruction scores
        val_re, test_re, val_re_full, test_re_full = self.train_vae(train_norm, val_norm, test_abs)

        if self.root_cause:
            self.val_re_full = val_re_full
            self.test_re_full = test_re_full

        # Set anomaly threshold (95th percentile of validation scores)
        threshold = np.quantile(val_re, 0.95)
        realtime_indicator = test_re
        anomaly_prediction = test_re > threshold

        # Calculate variable contributions to first anomaly
        variable_contributions = None
        first_anomaly_idx = next((i for i, pred in enumerate(anomaly_prediction) if pred), None)
        if first_anomaly_idx is not None:
            contribution = test_re_full[first_anomaly_idx]
            total = contribution.sum()
            variable_contributions = contribution / total if total != 0 else np.zeros_like(contribution)

        return realtime_indicator, anomaly_prediction, variable_contributions

    @staticmethod
    def normalize_errors(error_mat: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """Normalize error matrix using specified method.
        
        Args:
            error_mat (np.ndarray): Matrix of errors (samples x features)
            method (str): 'iqr' for interquartile range or 'z-score'
            
        Returns:
            norm_error (np.ndarray): Normalized error matrix
        """
        if method == 'iqr':
            median = np.median(error_mat, axis=0)
            q1 = np.quantile(error_mat, 0.25, axis=0)
            q3 = np.quantile(error_mat, 0.75, axis=0)
            iqr = q3 - q1 + 1e-8  # Add epsilon to avoid division by zero
            return (error_mat - median) / iqr
        elif method == 'z-score':
            scaler = StandardScaler()
            return scaler.fit_transform(error_mat)
        else:
            raise ValueError("Method must be 'iqr' or 'z-score'")

# Utility functions
def sliding_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply sliding window to time series data.
    
    Args:
        data (np.ndarray): Input data (time x features)
        window_size (int): Window length
        
    Returns:
        windowed_data (np.ndarray): Data with shape (time-window_size+1, window_size, features)
    """
    return data[np.arange(window_size)[None, :] + np.arange(len(data) - window_size + 1)[:, None]]

def load_data(save_folder: str = 'save') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed data from files."""
    train_obs = np.load(os.path.join(save_folder, 'train_label_0.npy'))
    train_forecast = np.load(os.path.join(save_folder, 'train_pred_0.npy'))
    val_obs = np.load(os.path.join(save_folder, 'val_label_0.npy'))
    val_forecast = np.load(os.path.join(save_folder, 'val_pred_0.npy'))
    test_obs = np.load(os.path.join(save_folder, 'test_label_0.npy'))
    test_forecast = np.load(os.path.join(save_folder, 'test_pred_0.npy'))
    return train_obs, train_forecast, val_obs, val_forecast, test_obs, test_forecast

# Main execution
def main():
    # Load dataset
    train_obs, train_forecast, val_obs, val_forecast, test_obs, test_forecast = load_data()
    
    # Initialize detector
    detector = AnomalyDetector(
        train_obs=train_obs,
        val_obs=val_obs,
        test_obs=test_obs,
        train_forecast=train_forecast,
        val_forecast=val_forecast,
        test_forecast=test_forecast,
        batch_size=128
    )
    
    # Run anomaly detection
    scores, predictions, contributions = detector.calculate_scores()
    
    # Print results
    print(f"Detected {np.sum(predictions)} anomalies in test set")
    if contributions is not None:
        top_vars = np.argsort(contributions)[::-1][:5]
        print("\nTop contributing variables to first anomaly:")
        for var_idx in top_vars:
            print(f"Variable {var_idx}: Contribution = {contributions[var_idx]:.3f}")

if __name__ == "__main__":
    main()