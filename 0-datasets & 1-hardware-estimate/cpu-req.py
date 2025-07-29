import torch
import torch.nn as nn
from torchinfo import summary

class DVQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim):
        super(DVQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        codebook = self.embedding.weight
        distances = torch.cdist(encoded, codebook)
        indices = torch.argmin(distances, dim=-1)
        quantized = self.embedding(indices)
        decoded = self.decoder(quantized)
        return quantized, decoded

class MHABiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_classes):
        super(MHABiGRU, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        self.gru_layers = nn.ModuleList()
        self.gru_layers.append(nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True))
        for _ in range(5):
            self.gru_layers.append(nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True))
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        for gru in self.gru_layers:
            attn_output, _ = gru(attn_output)
        last_hidden_state = attn_output[:, -1, :]
        x = torch.relu(self.fc1(last_hidden_state))
        x = self.fc2(x)
        return x

batch_size = 32
seq_length = 1000

dvq_80 = DVQVAE(input_dim=80, hidden_dim=128, num_embeddings=512, embedding_dim=64)
summary(dvq_80, input_size=(batch_size * seq_length, 80), col_names=["input_size", "output_size", "num_params", "mult_adds"])

dvq_10 = DVQVAE(input_dim=10, hidden_dim=64, num_embeddings=256, embedding_dim=32)
summary(dvq_10, input_size=(batch_size * seq_length, 10), col_names=["input_size", "output_size", "num_params", "mult_adds"])

mha_64 = MHABiGRU(input_dim=64, hidden_dim=32, num_heads=4, num_classes=2)
summary(mha_64, input_size=(batch_size, seq_length, 64), col_names=["input_size", "output_size", "num_params", "mult_adds"])

mha_10 = MHABiGRU(input_dim=10, hidden_dim=16, num_heads=2, num_classes=2)
summary(mha_10, input_size=(batch_size, seq_length, 10), col_names=["input_size", "output_size", "num_params", "mult_adds"])
