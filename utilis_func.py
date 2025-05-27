import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
from minepy import MINE  # MINE library for MIC calculation

# ======================================
# MIC Calculation and Visualization
# ======================================
def get_MIC(in_data: pd.DataFrame, threehold: float = 0.9, draw: bool = True) -> np.ndarray:
    """
    Calculate Mutual Information Coefficient (MIC) matrix for input data.
    
    Parameters:
    - in_data: Input data (pandas DataFrame)
    - threehold: Threshold for heatmap visualization (default: 0.9)
    - draw: Whether to plot heatmap (default: True)
    
    Returns:
    - MIC matrix (numpy array)
    """
    n = in_data.shape[1]
    MIC = np.zeros((n, n))
    mine = MINE(alpha=0.6, c=15)  # Initialize MINE instance
    
    # Iterate over all feature pairs
    for i in tqdm(np.arange(n), desc="Calculating MIC"):
        for j in np.arange(n):
            mine.compute_score(in_data.iloc[:, i].values, in_data.iloc[:, j].values)
            MIC[i, j] = mine.mic()
    
    # Plot heatmap if requested
    if draw:
        plt.figure(figsize=(6, 5))
        sns.heatmap(MIC, square=True, cmap='coolwarm')
        plt.title("Mutual Information Coefficient (MIC) Matrix")
        plt.show()
    
    return MIC


# ======================================
# Data Thresholding
# ======================================
def screen(data: np.ndarray, num: int) -> np.ndarray:
    """
    Threshold data by keeping top 'num' values and setting others to zero.
    
    Parameters:
    - data: Input data array
    - num: Number of top values to retain
    
    Returns:
    - Thresholded data array
    """
    data_copy = data.copy()
    sorted_indices = np.argsort(data_copy)[::-1]  # Descending order indices
    data_sorted = data_copy[sorted_indices]
    
    threshold = data_sorted[num-1]  # Get the num-th largest value as threshold
    data_copy[data_copy < threshold] = 0  # Set values below threshold to zero
    
    return data_copy


# ======================================
# Time Series Data Generation
# ======================================
def data_generate(data: pd.DataFrame, window_size: int, test_ratio: float, val_ratio: float):
    """
    Generate time-series data with sliding windows for training/validation/testing.
    
    Parameters:
    - data: Input time-series data (pandas DataFrame)
    - window_size: Size of historical window (t-p to t)
    - test_ratio: Proportion of data for testing
    - val_ratio: Proportion of training data for validation
    
    Returns:
    - Tuple of (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    x_offsets = np.sort(np.concatenate((np.arange(-(window_size-1), 1, 1),)))  # [-window_size+1, ..., 0]
    y_offsets = np.sort(np.arange(1, 2, 1))  # [1] (next time step)
    
    num_samples, num_nodes = data.shape
    data = data.values  # Convert to numpy array
    
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))  # Exclusive upper bound
    
    # Generate windowed data
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    
    # Split data into train/val/test
    num_samples_total = x.shape[0]
    num_test = int(num_samples_total * test_ratio)
    train_samples = num_samples_total - num_test
    num_train = int(train_samples * (1 - val_ratio))
    num_val = train_samples - num_train
    
    # Create splits
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    x_test, y_test = (
        x[-num_test:] if num_test > 0 else np.zeros((0, window_size, num_nodes)),
        y[-num_test:] if num_test > 0 else np.zeros((0, 1, num_nodes)),
    )
    
    return (x_train, y_train, x_val, y_val, x_test, y_test)


# ======================================
# Data Loading and Normalization
# ======================================
def normal_std(x):
    """Calculate normalized standard deviation"""
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class DataLoaderS(object):
    """
    Data loader for static data normalization and batching.
    
    Attributes:
    - P (int): Window size
    - h (int): Horizon
    - rawdat (np.ndarray): Raw data
    - dat (np.ndarray): Normalized data
    - n, m (int): Number of samples and features
    - scale (np.ndarray): Normalization scale
    - train/valid/test (tuple): Data splits
    """
    def __init__(self, file_name: str, train: float, valid: float, device: torch.device, horizon: int, window: int, normalize: int = 2):
        self.P = window
        self.h = horizon
        self.device = device
        
        # Load and normalize data
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros_like(self.rawdat)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        
        # Split data
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        
        # Convert scale to tensor
        self.scale = torch.from_numpy(self.scale).float().to(device)
        self.scale = Variable(self.scale)
        
        # Calculate metrics for evaluation
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.rse = normal_std(tmp.numpy())
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp))).item()
    
    def _normalized(self, normalize: int):
        """Internal method for data normalization"""
        if normalize == 0:
            self.dat = self.rawdat
        elif normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)
        elif normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / self.scale[i]
    
    def _split(self, train: int, valid: int, test: int):
        """Split data into training/validation/testing sets"""
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
    
    def _batchify(self, idx_set: list, horizon: int) -> tuple:
        """Convert index set to batched data"""
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - horizon + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X.to(self.device), Y.to(self.device)]
    
    def get_batches(self, inputs: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool = True):
        """Generate batched data iterator"""
        length = len(inputs)
        index = torch.randperm(length) if shuffle else torch.arange(length)
        
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            yield Variable(inputs[excerpt]), Variable(targets[excerpt])
            start_idx += batch_size


# ======================================
# Dynamic Data Loader
# ======================================
class DataLoaderM(object):
    """
    Dynamic data loader with padding and batching.
    
    Attributes:
    - batch_size (int): Batch size
    - current_ind (int): Current batch index
    - size (int): Total number of samples
    - num_batch (int): Total number of batches
    - xs/ys (np.ndarray): Input and target data
    """
    def __init__(self, xs: np.ndarray, ys: np.ndarray, batch_size: int, pad_with_last_sample: bool = True):
        self.batch_size = batch_size
        self.current_ind = 0
        
        # Pad with last sample to make data divisible by batch size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        
        self.size = len(xs)
        self.num_batch = self.size // batch_size
        self.xs = xs
        self.ys = ys
    
    def shuffle(self):
        """Shuffle the dataset"""
        permutation = np.random.permutation(self.size)
        self.xs, self.ys = self.xs[permutation], self.ys[permutation]
    
    def get_iterator(self):
        """Generate data iterator"""
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_idx = self.batch_size * self.current_ind
                end_idx = min(self.size, self.batch_size * (self.current_ind + 1))
                yield (self.xs[start_idx:end_idx, ...], self.ys[start_idx:end_idx, ...])
                self.current_ind += 1
        return _wrapper()


# ======================================
# Data Scaling Utilities
# ======================================
class StandardScaler:
    """Standard scaler for data normalization"""
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean


# ======================================
# Graph Adjacency Matrix Utilities
# ======================================
def sym_adj(adj: sp.spmatrix) -> np.ndarray:
    """Symmetrically normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj: sp.spmatrix) -> np.ndarray:
    """Asymmetrically normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj: sp.spmatrix) -> sp.spmatrix:
    """Calculate normalized graph Laplacian"""
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: float = 2, undirected: bool = True) -> np.ndarray:
    """Calculate scaled graph Laplacian"""
    if undirected:
        adj_mx = np.maximum(adj_mx, adj_mx.T)
    L = calculate_normalized_laplacian(adj_mx)
    
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    
    L = sp.csr_matrix(L)
    M = L.shape[0]
    I = sp.identity(M, format='csr', dtype=L.dtype)
    return ((2 / lambda_max * L) - I).astype(np.float32).todense()


# ======================================
# Data I/O Utilities
# ======================================
def load_pickle(pickle_file: str):
    """Load pickle file with error handling"""
    try:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading {pickle_file}: {e}")
        raise

def load_adj(pkl_filename: str) -> np.ndarray:
    """Load adjacency matrix from pickle file"""
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj

def load_dataset(dataset_dir: str, batch_size: int, valid_batch_size: int = None, test_batch_size: int = None, scaling_required: bool = True) -> dict:
    """Load dataset from directory and create data loaders"""
    data = {}
    for category in ['train', 'val', 'test']:
        file_path = os.path.join(dataset_dir, f"{category}.npz")
        with np.load(file_path) as cat_data:
            data[f'x_{category}'] = cat_data['x']
            data[f'y_{category}'] = cat_data['y']
    
    # Initialize scaler
    scaler = StandardScaler(
        mean=data['x_train'][..., 0].mean(),
        std=data['x_train'][..., 0].std()
    )
    
    # Apply scaling if needed
    if scaling_required:
        for category in ['train', 'val', 'test']:
            data[f'x_{category}'][..., 0] = scaler.transform(data[f'x_{category}'][..., 0])
    
    # Create data loaders
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size or batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size or batch_size)
    data['scaler'] = scaler
    
    return data


# ======================================
# Evaluation Metrics
# ======================================
def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Calculate masked Mean Squared Error"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val).float()
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = (preds - labels)**2 * mask
    return torch.where(torch.isnan(loss), torch.zeros_like(loss), loss).mean()

def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Calculate masked Root Mean Squared Error"""
    return torch.sqrt(masked_mse(preds, labels, null_val))

def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Calculate masked Mean Absolute Error"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val).float()
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels) * mask
    return torch.where(torch.isnan(loss), torch.zeros_like(loss), loss).mean()

def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Calculate masked Mean Absolute Percentage Error"""
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val).float()
    
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    loss = torch.abs(preds - labels) / labels * mask
    return torch.where(torch.isnan(loss), torch.zeros_like(loss), loss).mean()


def metric(pred: torch.Tensor, real: torch.Tensor) -> tuple:
    """Calculate evaluation metrics"""
    mae = None  # Removed to match original code
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


# ======================================
# Node Feature Utilities
# ======================================
def load_node_feature(path: str) -> torch.Tensor:
    """Load and normalize node features"""
    with open(path) as fi:
        x = []
        for line in fi:
            line = line.strip().split(",")
            features = [float(t) for t in line[1:]]
            x.append(features)
    
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return torch.tensor((x - mean) / std, dtype=torch.float)


# ======================================
# Causal Adjacency Matrix Utilities
# ======================================
def get_Causal_adjm(file_path: str, num_nodes: int, idx: list = None, flag: int = 1) -> torch.Tensor:
    """
    Load causal adjacency matrix from file.
    
    Parameters:
    - file_path: Path to adjacency matrix file
    - num_nodes: Number of nodes
    - idx: Optional list of indices to subset
    - flag: 1 for edge list format, 0 for CSV format
    
    Returns:
    - Adjacency matrix as torch tensor
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    if flag == 1:
        # Load from edge list file
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                
                if len(parts) != 2:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                
                try:
                    node1 = int(parts[0]) - 1
                    node2 = int(float(parts[1]))
                    
                    if 0 <= node1 < num_nodes and 0 <= node2 < num_nodes:
                        adj_matrix[node1, node2] = 1
                    else:
                        print(f"Node index out of range: {node1}, {node2}")
                except ValueError:
                    print(f"Error parsing line: {line.strip()}")
    else:
        # Load from CSV file
        labels_df = pd.read_csv(file_path)
        labels_df.fillna(0, inplace=True)
        adj_matrix = labels_df.values
    
    # Subset matrix if indices provided
    if idx is not None:
        adj_matrix = adj_matrix[idx, :][:, idx]
    
    return torch.tensor(adj_matrix)