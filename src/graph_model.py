import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_mediapipe_skeleton(num_nodes=33):
    """
    Returns the adjacency matrix for MediaPipe's 33-point skeleton.
    """
    # MediaPipe Pose Topology
    # 0-10: Face, 11-12: Shoulders, 13-16: Arms, 23-24: Hips, 25-32: Legs/Feet
    edges = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Arms
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right
        # Legs
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)  # Right
    ]

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # Self-loops and Normalization
    np.fill_diagonal(A, 1)
    rowsum = A.sum(axis=1) + 1e-6
    D_inv = np.diag(rowsum ** -1)
    A_norm = D_inv.dot(A)

    return torch.from_numpy(A_norm).float()


class GraphSpatialEncoder(nn.Module):
    def __init__(self, num_nodes=33, input_dim=3, hidden_dim=64, use_mask=False):
        super(GraphSpatialEncoder, self).__init__()

        self.num_nodes = num_nodes
        self.use_mask = use_mask

        # 1. Geometry with Self-Loops (A + I)
        fixed_adj = get_mediapipe_skeleton(num_nodes)
        fixed_adj = fixed_adj + torch.eye(num_nodes)
        self.register_buffer('adj', fixed_adj)

        if use_mask:
            # Multiplicative mask initialized to 1
            self.edge_mask = nn.Parameter(torch.ones(num_nodes, num_nodes))

        # 2. Graph Convolution Layers
        self.gcn_1 = nn.Linear(input_dim, 32)
        self.gcn_2 = nn.Linear(32, 64)

        # 3. Aggregation (Pooling instead of flattening)
        self.projection = nn.Linear(64, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()  # Swapped from Tanh

    def forward(self, x):
        batch, seq_len, nodes, coords = x.shape
        x = x.reshape(batch * seq_len, nodes, coords)

        # Apply learnable edge weighting
        curr_adj = self.adj * self.edge_mask if self.use_mask else self.adj

        # --- Layer 1 ---
        support = torch.matmul(curr_adj, x)
        x = self.relu(self.gcn_1(support))

        # --- Layer 2 ---
        support = torch.matmul(curr_adj, x)
        x = self.relu(self.gcn_2(support))

        # --- Aggregation (Global Average Pooling) ---
        x = torch.mean(x, dim=1)  # Shape: (B*T, 64)

        x = self.projection(x)
        x = self.dropout(x)

        return x.view(batch, seq_len, -1)


class LandmarkPredictor(nn.Module):
    def __init__(self, num_nodes=33, input_dim=3, num_layers=1, embedding_dim=128, lstm_hidden=256):
        super(LandmarkPredictor, self).__init__()

        # 1. Spatial Encoder (The Graph Part)
        self.spatial_encoder = GraphSpatialEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=embedding_dim,  # This is the input size for LSTM
            use_mask=True
        )

        # 2. Temporal Encoder (The LSTM Part)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        # 3. Final Prediction Head
        # Projects LSTM output back to 33 landmarks
        self.head = nn.Linear(lstm_hidden * 2, num_nodes * input_dim)

        self.act = nn.Tanh()

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        x: (Batch, Time, 33, 3)
        """
        batch_size, seq_len, nodes, dims = x.shape

        # --- 1. Get Frame Embeddings ---
        # This gives you the [Batch, Frame, Hidden] tensor you asked for
        spatial_features = self.spatial_encoder(x)

        # --- 2. LSTM Processing ---
        lstm_out, (h_n, c_n) = self.lstm(spatial_features)

        predictions = self.head(lstm_out[:, -1, :])

        predictions = self.dropout(predictions)

        predictions = self.act(predictions)

        return predictions
