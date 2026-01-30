"""
Transformer Model for Lap Sequence Prediction
Treats laps as sequence tokens with positional encoding
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class LapSequenceTransformer(nn.Module):
    """
    Transformer model treating laps as sequence tokens
    Predicts performance degradation from lap time patterns
    """
    
    def __init__(
        self,
        input_dim=10,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_length=50,
        dropout=0.1,
        output_dim=1
    ):
        super(LapSequenceTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.Sigmoid()
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_length, :].to(x.device)
        x = x + pos_enc
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Use last lap
        last_lap = encoded[:, -1, :]
        
        # Prediction
        output = self.fc_out(last_lap)
        
        return output


class LapSequenceDataset(Dataset):
    """
    Dataset for lap sequence data
    Creates sequences of N laps to predict degradation
    """
    
    def __init__(self, df, sequence_length=10, features=None, target='IsDegraded'):
        from sklearn.preprocessing import StandardScaler
        
        self.sequence_length = sequence_length
        self.target = target
        
        if features is None:
            self.features = [
                'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                'LapTime_RollingStd', 'LapTime_CV', 'SectorInconsistency_Score',
                'StintProgress_Pct', 'DeltaToPersonalBest', 'TyreLife'
            ]
        else:
            self.features = features
        
        self.sequences = []
        self.labels = []
        
        # Create sliding window sequences
        for (driver, race), group in df.groupby(['Driver', 'RaceNumber']):
            group = group.sort_values('LapNumber').reset_index(drop=True)
            
            for i in range(len(group) - sequence_length):
                seq = group.iloc[i:i+sequence_length][self.features].values
                label = group.iloc[i+sequence_length][target]
                
                self.sequences.append(seq)
                self.labels.append(label)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # Normalize sequences
        self.scaler = StandardScaler()
        n_samples, n_timesteps, n_features = self.sequences.shape
        self.sequences = self.sequences.reshape(-1, n_features)
        self.sequences = self.scaler.fit_transform(self.sequences)
        self.sequences = self.sequences.reshape(n_samples, n_timesteps, n_features)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])


if __name__ == "__main__":
    # Test model
    print("Testing Transformer Model...")
    
    model = LapSequenceTransformer(
        input_dim=10,
        d_model=64,
        nhead=4,
        num_layers=2
    )
    
    # Dummy input
    batch_size = 8
    seq_length = 10
    input_dim = 10
    
    x = torch.randn(batch_size, seq_length, input_dim)
    output = model(x)
    
    print(f"✅ Model output shape: {output.shape}")
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model test passed!")