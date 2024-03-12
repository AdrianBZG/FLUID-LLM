import torch.nn as nn


class PositionalEmbeddings(nn.Module):
    """Positional embeddings layer, for time, x and y."""

    def __init__(self, hidden_size, max_pos_embeddings):
        super().__init__()
        self.x_embeddings = nn.Embedding(max_pos_embeddings, hidden_size)
        self.y_embeddings = nn.Embedding(max_pos_embeddings, hidden_size)
        self.time_embeddings = nn.Embedding(max_pos_embeddings, hidden_size)

    def forward(self, position_ids):
        x_embeddings = self.x_embeddings(position_ids[..., 0])
        y_embeddings = self.y_embeddings(position_ids[..., 1])
        time_embeddings = self.time_embeddings(position_ids[..., 2])

        return x_embeddings + y_embeddings + time_embeddings
