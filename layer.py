import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class nconv(nn.Module):
    """Element-wise multiplication with adjacency matrix"""
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
            A: Adjacency matrix (shape: [node, node])
        Returns:
            Output tensor after convolution (shape: [batch, channel, node, length])
        """
        return torch.einsum('ncwl,vw->ncvl', (x, A)).contiguous()

class dy_nconv(nn.Module):
    """Dynamic element-wise multiplication with adjacency matrix"""
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
            A: Dynamic adjacency matrix (shape: [batch, node, node, length])
        Returns:
            Output tensor after dynamic convolution (shape: [batch, channel, node, length])
        """
        return torch.einsum('ncvl,nvwl->ncwl', (x, A)).contiguous()

class linear(nn.Module):
    """1x1 Convolution for linear transformation"""
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
        Returns:
            Output tensor after linear transformation (shape: [batch, channel, node, length])
        """
        return self.mlp(x)

class prop(nn.Module):
    """Graph Propagation Layer"""
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep  # Propagation depth
        self.dropout = dropout
        self.alpha = alpha  # Weight for residual connection

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
            adj: Adjacency matrix (shape: [node, node])
        Returns:
            Output tensor after graph propagation (shape: [batch, channel, node, length])
        """
        adj = adj + torch.eye(adj.size(0), device=x.device)  # Add self-loop
        d = adj.sum(1)[:, None, None]  # Degree matrix
        a = adj / d  # Normalized adjacency
        h = x
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        return self.mlp(h)

class mixprop(nn.Module):
    """Mixed Propagation Layer with Multi-Level Aggregation"""
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
            adj: Adjacency matrix (shape: [node, node])
        Returns:
            Output tensor after multi-level propagation (shape: [batch, channel, node, length])
        """
        adj = adj + torch.eye(adj.size(0), device=x.device)
        d = adj.sum(1)[:, None, None]
        a = adj / d
        h = x
        out = [h]  # Collect outputs at each propagation step
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)  # Concatenate multi-level features
        return self.mlp(ho)

class dy_mixprop(nn.Module):
    """Dynamic Mixed Propagation Layer with Learnable Adjacency"""
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
        Returns:
            Output tensor after bidirectional dynamic propagation (shape: [batch, channel, node, length])
        """
        x1 = torch.tanh(self.lin1(x))  # [batch, c_in, node, length]
        x2 = torch.tanh(self.lin2(x))  # [batch, c_in, node, length]
        adj = self.nconv(x1.transpose(2, 1), x2)  # Compute dynamic adjacency [batch, node, node, length]
        adj0 = torch.softmax(adj, dim=2)  # Forward direction
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)  # Backward direction

        # Forward propagation
        h_forward = x
        out_forward = [h_forward]
        for _ in range(self.gdep):
            h_forward = self.alpha * x + (1 - self.alpha) * self.nconv(h_forward, adj0)
            out_forward.append(h_forward)
        ho1 = self.mlp1(torch.cat(out_forward, dim=1))

        # Backward propagation
        h_backward = x
        out_backward = [h_backward]
        for _ in range(self.gdep):
            h_backward = self.alpha * x + (1 - self.alpha) * self.nconv(h_backward, adj1)
            out_backward.append(h_backward)
        ho2 = self.mlp2(torch.cat(out_backward, dim=1))

        return ho1 + ho2  # Combine bidirectional outputs

class dilated_1D(nn.Module):
    """1D Dilated Convolution Layer"""
    def __init__(self, cin: int, cout: int, dilation_factor: int = 2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.Conv2d(cin, cout, kernel_size=(1, 7), dilation=(1, dilation_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
        Returns:
            Output tensor after dilated convolution (shape: [batch, channel, node, length])
        """
        return self.tconv(x)

class dilated_inception(nn.Module):
    """Inception-style Dilated Convolution Module"""
    def __init__(self, cin: int, cout: int, dilation_factor: int = 2):
        super(dilated_inception, self).__init__()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.ModuleList([
            nn.Conv2d(cin, int(cout/len(self.kernel_set)), (1, kern), dilation=(1, dilation_factor))
            for kern in self.kernel_set
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
        Returns:
            Output tensor after multi-kernel dilated convolution (shape: [batch, channel, node, length])
        """
        outputs = []
        for conv in self.tconv:
            outputs.append(conv(x))
        # Align sequence lengths
        max_len = outputs[-1].shape[-1]
        outputs = [out[..., :max_len] for out in outputs]
        return torch.cat(outputs, dim=1)

class LayerNorm(nn.Module):
    """Layer Normalization with Channel-wise Parameters"""
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight, self.bias = None, None
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, length])
            idx: Channel indices for parameter selection (optional)
        Returns:
            Normalized tensor (shape: [batch, channel, node, length])
        """
        if self.elementwise_affine:
            weight = self.weight[:, idx, :] if idx is not None else self.weight
            bias = self.bias[:, idx, :] if idx is not None else self.bias
            return F.layer_norm(x, x.shape[1:], weight, bias, self.eps)
        else:
            return F.layer_norm(x, x.shape[1:], self.weight, self.bias, self.eps)

class GATLayer(nn.Module):
    """Single-Head Graph Attention Layer"""
    def __init__(self, feature_dim: int, dropout: float = 0.2, alpha: float = 0.1, concat: bool = False):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = feature_dim
        self.alpha = alpha
        self.concat = concat
        self.a1 = nn.Parameter(init.xavier_uniform_(torch.zeros(1, 1, feature_dim, 1), gain=1.414))
        self.a2 = nn.Parameter(init.xavier_uniform_(torch.zeros(1, 1, feature_dim, 1), gain=1.414))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, channel, node, feature])
            adj: Adjacency matrix (shape: [node, node])
        Returns:
            Output tensor after attention (shape: [batch, channel, node, feature])
        """
        h = x
        f_1 = torch.matmul(h, self.a1).squeeze(-1)  # [batch, channel, node]
        f_2 = torch.matmul(h, self.a2).squeeze(-1)  # [batch, channel, node]
        e = self.leakyrelu(f_1.unsqueeze(-1) + f_2.unsqueeze(-2))  # [batch, channel, node, node]

        # Mask with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0).unsqueeze(0) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        return torch.matmul(attention, h)  # [batch, channel, node, feature]

class MHGATLayer(nn.Module):
    """Multi-Head Graph Attention Layer"""
    def __init__(self, feature_dim: int, num_heads: int = 16, dropout: float = 0.2, alpha: float = 0.1, concat: bool = False):
        super(MHGATLayer, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.alpha = alpha
        self.W = nn.Parameter(init.xavier_uniform_(
            torch.zeros(num_heads, feature_dim, feature_dim), gain=1.414
        ))
        self.a1 = nn.Parameter(init.xavier_uniform_(
            torch.zeros(num_heads, feature_dim, 1), gain=1.414
        ))
        self.a2 = nn.Parameter(init.xavier_uniform_(
            torch.zeros(num_heads, feature_dim, 1), gain=1.414
        ))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attention_per_head = None
        self.attention_aggregated = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, head, node, feature])
            adj: Adjacency matrix (shape: [node, node])
        Returns:
            Output tensor after multi-head attention (shape: [batch, head, node, feature])
        """
        # Linear transformation
        h = torch.einsum('bhnf, hfd -> bhnd', x, self.W)  # [batch, head, node, feature]

        # Attention calculation
        f_1 = torch.matmul(h, self.a1).squeeze(-1)  # [batch, head, node]
        f_2 = torch.matmul(h, self.a2).squeeze(-1)  # [batch, head, node]
        e = self.leakyrelu(f_1.unsqueeze(-1) + f_2.unsqueeze(-2))  # [batch, head, node, node]

        # Mask and normalize attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(0).unsqueeze(1) > 0, e, zero_vec)
        attention = F.softmax(attention / 0.0005, dim=-1)  # Temperature scaling
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Save attention maps for visualization
        self.attention_per_head = attention
        self.attention_aggregated = attention.mean(dim=(0, 1))

        # Compute output
        h_prime = torch.matmul(attention, h)  # [batch, head, node, feature]
        return F.elu(h_prime)

    def get_attention_maps(self):
        """Return attention maps for visualization"""
        return self.attention_per_head, self.attention_aggregated

class GAT(nn.Module):
    """Multi-Head Graph Attention Network"""
    def __init__(self, feature_dim: int, c_in: int, c_out: int, num_layers: int, dropout: float = 0.2, alpha: float = 0.1):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList([
            MHGATLayer(feature_dim, c_in, dropout, alpha) for _ in range(num_layers)
        ])
        self.mlp = linear((num_layers + 1) * c_in, c_out)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [batch, head, node, feature])
            adj: Adjacency matrix (shape: [node, node])
        Returns:
            Output tensor after multi-layer attention (shape: [batch, channel, node, length])
        """
        layer_outputs = [x]
        for layer in self.layers:
            x = layer(x, adj)
            layer_outputs.append(x)
        ho = torch.cat(layer_outputs, dim=1)
        return self.mlp(ho)

# ------------------------------
# Fourier Amplitude Network (FAN) Layer
# ------------------------------
class FANLayer(nn.Module):
    """
    FANLayer: Encodes input features into Fourier domain using cosine/sine and learns residual features.
    
    Args:
        input_dim (int): Number of input features.
        output_dim (int): Total number of output features.
        p_ratio (float): Ratio of features allocated to Fourier components (cos+sin).
        activation (str): Activation function for residual component ('gelu'/'relu' etc.).
    """
    def __init__(self, input_dim: int, output_dim: int, p_ratio: float = 0.25, activation: str = 'gelu'):
        super(FANLayer, self).__init__()
        assert 0 < p_ratio < 0.5, "p_ratio must be between (0, 0.5)"
        
        self.p_dim = int(output_dim * p_ratio)  # Dim for each of cos/sin
        self.g_dim = output_dim - 2 * self.p_dim  # Dim for residual component
        
        # Fourier component transformation
        self.fc_p = nn.Linear(input_dim, self.p_dim, bias=True)
        
        # Residual component transformation
        self.fc_g = nn.Linear(input_dim, self.g_dim, bias=True)
        
        # Activation function
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (shape: [B, C, N, F] - Batch, Channel, Node, Feature)
        Returns:
            Output tensor (shape: [B, C, N, output_dim])
        """
        B, C, N, F = x.shape
        x_flat = x.view(-1, F)  # Flatten to [B*C*N, F]
        
        # Fourier components (cos and sin)
        p = self.fc_p(x_flat)
        x_fourier = torch.cat([torch.cos(p), torch.sin(p)], dim=-1)  # [B*C*N, 2*p_dim]
        
        # Residual component
        g = self.activation(self.fc_g(x_flat))  # [B*C*N, g_dim]
        
        # Concatenate and reshape
        x_out = torch.cat([x_fourier, g], dim=-1).view(B, C, N, -1)
        return x_out

# ------------------------------
# Gated Weighting Layer
# ------------------------------
class GatedWeightingLayer(nn.Module):
    """
    Gated fusion of original features and FAN features.
    
    Args:
        input_dim (int): Dimension of input features (must match FAN output_dim).
    """
    def __init__(self, input_dim: int):
        super(GatedWeightingLayer, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # Concatenate input and FAN output
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, fan_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Original features (shape: [B, C, N, F])
            fan_x: FAN-transformed features (shape: [B, C, N, F])
        Returns:
            Gated output (shape: [B, C, N, F])
        """
        B, C, N, F = x.shape
        x_flat = x.view(B, C*N, F)
        fan_flat = fan_x.view(B, C*N, F)
        
        # Gate mechanism
        concat = torch.cat([x_flat, fan_flat], dim=-1)  # [B, C*N, 2F]
        g = self.gate(concat)  # [B, C*N, F]
        g = g.view(B, C, N, F)
        
        return g * x + (1 - g) * fan_x  # Adaptive feature fusion

# ------------------------------
# Utility Functions
# ------------------------------
def init_weights(module: nn.Module):
    """Initialize weights for nn.Module"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            init.zeros_(module.bias.data)
