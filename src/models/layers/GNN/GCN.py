import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


class GCN_layers(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """
        Initializes the Graph Convolutional Network with variable input dimensions,
        output dimensions, and hidden layers.

        Args:
        input_dim (int): Dimensionality of the input features.
        hidden_dims (list of int): List containing the sizes of each hidden layer.
        output_dim (int): Dimensionality of the output features.
        """
        super().__init__()

        # Create a list of all GCN convolutional layers
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.out_conv = GATv2Conv(input_dim, output_dim)
        else:
            self.convs.append(GATv2Conv(input_dim, hidden_dim))
            for l in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim))

            # Output layer
            self.out_conv = GATv2Conv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN.

        Args:
        x (Tensor): Node feature matrix (shape [num_nodes, input_dim])
        edge_index (Tensor): Graph connectivity in COO format (shape [2, num_edges])
        batch (Tensor): Batch vector, which assigns each node to a specific example

        Returns:
        Tensor: Output from the GNN after applying log softmax.
        """
        # Pass through each convolutional layer with ReLU activation
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, p=0.5, training=self.training)

        # Pass through the output convolutional layer
        x = self.out_conv(x, edge_index)
        return x

