import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any
from minepy import MINE

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, window_size: int, target_idx: int):
        self.data = data
        self.window_size = window_size
        self.target_idx = target_idx
        
    def __len__(self) -> int:
        return len(self.data) - self.window_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx+self.window_size, :]
        y = self.data[idx+self.window_size, self.target_idx:self.target_idx+1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)
        context = context.squeeze(1)
        output = self.fc(context)
        
        return output, attn_weights.squeeze(2)

def create_subfolder(output_folder: str, subfolder_name: str) -> str:
    """Create subfolder and return path"""
    subfolder_path = os.path.join(output_folder, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        print(f"Created subfolder '{subfolder_name}'")
    return subfolder_path

def load_and_preprocess_data(data_name: str) -> pd.DataFrame:
    """Load and preprocess data"""
    print(f"Loading data: {data_name}")
    data = pd.read_csv(data_name, header=None, sep='\s+')
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return pd.DataFrame(data_scaled, columns=data.columns)

def get_MIC(data: pd.DataFrame) -> np.ndarray:
    """Calculate Maximum Information Coefficient (MIC) matrix"""

    n = data.shape[1]
    MIC = np.zeros((n, n))
    
    print("Calculating MIC matrix...")
    for i in tqdm(range(n)):
        for j in range(n):
            if i != j:
                mine = MINE(alpha=0.6, c=15)
                mine.compute_score(data.iloc[:, i], data.iloc[:, j])
                MIC[i, j] = mine.mic()
    
    return MIC

def screen(mic_values: np.ndarray, num_nodes: int) -> np.ndarray:
    sorted_indices = np.argsort(mic_values)[::-1]
    
    mask = np.zeros_like(mic_values, dtype=bool)
    mask[sorted_indices[:num_nodes]] = True
    
    return mask

def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, num_epochs: int, device: torch.device) -> Dict[str, Any]:
    model.train()
    history = {'loss': []}
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for batch_idx, (data, target) in progress_bar:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output, attention = model(data)
            loss = criterion(output, target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_description(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return history

def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, List[float], List[float]]:
    model.eval()
    total_loss = 0
    actual_values = []
    predicted_values = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output, attention = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            actual_values.extend(target.cpu().numpy().flatten())
            predicted_values.extend(output.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, actual_values, predicted_values

def save_adjacency_matrix(adj_matrix: np.ndarray, output_path: str) -> None:
    """Save adjacency matrix to file"""
    # Save as CSV
    csv_path = f"{output_path}.csv"
    np.savetxt(csv_path, adj_matrix, delimiter=',', fmt='%d')
    print(f"Adjacency matrix saved to: {csv_path}")
    
    # Save as NPY
    npy_path = f"{output_path}.npy"
    np.save(npy_path, adj_matrix)
    print(f"Adjacency matrix saved to: {npy_path}")

def main():
    # Configuration parameters
    data_name = ''
    output_folder = ''  # Output folder name
    window_size = 20        # Window length for prediction
    num_nodes = 20          # Number of time series variables for prediction
    hidden_size = 128
    num_layers = 2
    output_size = 1
    num_epochs = 10
    learning_rate = 0.001
    dropout = 0.2
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    check_path = create_subfolder(output_folder, 'check_folder')
    selected_var_path = create_subfolder(output_folder, 'selected_var')
    var_shuffled_path = create_subfolder(output_folder, 'var_shuffled')
    
    data = load_and_preprocess_data(data_name)
    num_features = data.shape[1]  # Get actual number of features
    
    base_name, ext = os.path.splitext(data_name)
    mic_file_name = f'SCGI_MIC_{base_name}.npy'  # MIC file name related to SCGI
    
    if os.path.exists(mic_file_name):
        print(f"Loading SCGI MIC matrix: {mic_file_name}")
        MIC = np.load(mic_file_name)
    else:
        print("Calculating SCGI MIC matrix...")
        MIC = get_MIC(data)
        np.save(mic_file_name, MIC)
        print(f"SCGI MIC matrix saved to: {mic_file_name}")
    
    attention_matrix = np.zeros((num_features, num_features), dtype=int)
    shuffle_matrix = np.zeros((num_features, num_features), dtype=int)
    
    file_name_oc = os.path.join(output_folder, f'SCGI_selected_var_{hidden_size}_{num_layers}.txt')
    file_name_oc_s = os.path.join(output_folder, f'SCGI_selected_var_shuffer_{hidden_size}_{num_layers}.txt')
    
    # Process each variable
    for var_num in range(num_features):
        print(f"\nProcessing SCGI variable {var_num+1}/{num_features}")
        
        mask = screen(MIC[var_num, :], num_nodes)
        data_selected = data.iloc[:900+window_size, mask]
        
        train_size = 750
        val_size = 150
        
        dataset = TimeSeriesDataset(data_selected.values, window_size, np.where(mask)[0][0])
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
        )
        
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = LSTMModel(
            input_size=num_nodes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"Training SCGI model - Variable {var_num+1}")
        history = train_model(model, train_loader, criterion, optimizer, num_epochs, device)
        
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f'SCGI Validation Loss: {val_loss:.4f}')
        
        attention_weights = model.state_dict()["attention"].detach().cpu().numpy().flatten()
        print(f"SCGI Attention Parameters: {attention_weights}")
        
        selected_var_indices = np.where(attention_weights > 1)[0]
        selected_vars = [data_selected.columns[i] for i in selected_var_indices]
        print(f"Variables selected based on SCGI attention: {selected_vars}")
        
        for idx in selected_var_indices:
            related_var = data_selected.columns[idx]
            attention_matrix[var_num, related_var] = 1
        
        selected_var_shuffle = []
        
        for index in selected_var_indices:
            print(f"Testing importance of SCGI variable {data_selected.columns[index]}")
            
            # Create shuffled dataset
            data_shuffle = data_selected.copy()
            original_sequence = data_shuffle.iloc[:, index].values
            
            # Shuffle by window
            num_segments = len(original_sequence) // window_size
            if len(original_sequence) % window_size != 0:
                num_segments += 1
                
            split_sequence = np.array_split(original_sequence, num_segments)
            np.random.shuffle(split_sequence)
            shuffled_sequence = np.concatenate(split_sequence)
            
            data_shuffle.iloc[:, index] = shuffled_sequence
            
            dataset_shuffle = TimeSeriesDataset(data_shuffle.values, window_size, np.where(mask)[0][0])
            _, val_shuffle = torch.utils.data.random_split(
                dataset_shuffle, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
            )
            
            shuffle_loader = DataLoader(val_shuffle, batch_size=batch_size, shuffle=False)
            
            # Evaluate model performance after shuffling
            shuffle_loss, _, _ = evaluate_model(model, shuffle_loader, criterion, device)
            print(f"Loss after shuffling SCGI variable {data_selected.columns[index]}: {shuffle_loss:.4f}")
            
            # If performance drops significantly, consider the variable important
            if shuffle_loss / val_loss > 1.5:
                selected_var_shuffle.append(data_selected.columns[index])
                print(f"SCGI variable {data_selected.columns[index]} confirmed as important")
                shuffle_matrix[var_num, data_selected.columns[index]] = 1
        
        print(f"SCGI variable {var_num+1} processing completed")
    
    # Save adjacency matrices
    adj_matrix_path = os.path.join(output_folder, f'SCGI_adjacency_matrix_{hidden_size}_{num_layers}')
    save_adjacency_matrix(attention_matrix, f"{adj_matrix_path}_attention")
    save_adjacency_matrix(shuffle_matrix, f"{adj_matrix_path}_shuffle")
    
    print("Processing of all SCGI variables completed. Adjacency matrices saved.")

if __name__ == "__main__":
    main()    