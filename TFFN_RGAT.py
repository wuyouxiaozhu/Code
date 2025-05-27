from layer import *
from utilis_func import get_Causal_adjm

class STRGAT(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, 
             predefined_A=None, dropout=0.3, conv_channels=32, 
             residual_channels=32, skip_channels=64, end_channels=128, 
             seq_length=12, in_dim=2, out_dim=12, layers=3, 
             propalpha=0.05, layer_norm_affline=True):
        """
        Initialize the STRGAT model.

        Args:
            gcn_true (bool): Whether to use graph convolution.
            buildA_true (bool): Whether to build the adjacency matrix.
            gcn_depth (int): Depth of the graph convolution.
            num_nodes (int): Number of nodes in the graph.
            device (torch.device): Device to run the model on.
            predefined_A (torch.Tensor, optional): Predefined adjacency matrix. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.3.
            conv_channels (int, optional): Number of channels in the convolutional layers. Defaults to 32.
            residual_channels (int, optional): Number of channels in the residual connections. Defaults to 32.
            skip_channels (int, optional): Number of channels in the skip connections. Defaults to 64.
            end_channels (int, optional): Number of channels in the final convolutional layers. Defaults to 128.
            seq_length (int, optional): Length of the input sequence. Defaults to 12.
            in_dim (int, optional): Input dimension. Defaults to 2.
            out_dim (int, optional): Output dimension. Defaults to 12.
            layers (int, optional): Number of layers in the model. Defaults to 3.
            propalpha (float, optional): Propagation alpha. Defaults to 0.05.
            layer_norm_affline (bool, optional): Whether to use affine transformation in layer normalization. Defaults to True.
        """
        super(STRGAT, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.fan_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.seq_length = seq_length
        kernel_size = 7
        
        self.receptive_field = layers*(kernel_size-1) + 1
        self.g = nn.Parameter(torch.tensor(0.5))

        for i in range(1):
            
            rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                rf_size_j = rf_size_i+j*(kernel_size-1)
                # Append dilated inception layer for filters
                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                # Append dilated inception layer for gates
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                # Append convolutional layer for residual connections
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))
                # Append FAN layer
                self.fan_convs.append(FANLayer(input_dim=100-(j-1)*6, output_dim=100-j*6))
                if self.seq_length>self.receptive_field:
                    # Append skip connection convolutional layer
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    # Append skip connection convolutional layer
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    # Append GAT layer
                    self.gconv1.append(GAT(100-j*6, conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    # Append GAT layer
                    self.gconv2.append(GAT(100-j*6, conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    # Append layer normalization layer
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    # Append layer normalization layer
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= 1

        self.layers = layers
        # Final convolutional layer 1
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        # Final convolutional layer 2
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            # Skip connection at the beginning
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            # Skip connection at the end
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            # Skip connection at the beginning
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            # Skip connection at the end
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
            
        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        """
        Forward pass of the STRGAT model.

        Args:
            input (torch.Tensor): Input tensor.
            idx (torch.Tensor, optional): Index tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor, optional: Adjacency matrix if not in training mode.
        """
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                adp =  get_Causal_adjm("adp_new.csv", self.num_nodes, idx, 2)
            else:
                adp = self.predefined_A
        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            fan_x = self.fan_convs[i](x)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            
            g_value = torch.sigmoid(self.g)
            x = g_value * x + (1 - g_value) * fan_x
            
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                # x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1,0))
                x = self.gconv1[i](x, adp)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        if self.training:
            return x
        else:
            return x, adp