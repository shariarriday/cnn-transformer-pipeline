import torch
import torch.nn as nn
from torchvision import models
import math

class CNNFeatureExtractor(nn.Module):
	def __init__(self):
		super(CNNFeatureExtractor, self).__init__()
		self.cnn_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
		self.cnn_backbone.fc = nn.Identity()

	def forward(self, x):
		batch_size, num_frames, channels, height, width = x.shape
		x = x.view(batch_size * num_frames, channels, height, width)
		features = self.cnn_backbone(x)
		features = features.view(batch_size, num_frames, -1)
		return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Basic positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Learnable temporal components
        self.temporal_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.temporal_attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms for attention and MLP
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Learnable scale factor for positional encoding
        self.scale = nn.Parameter(torch.ones(1))
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        batch_size = x.size(0)
        
        # Add scaled positional encoding
        pos_encoding = self.pe[:seq_len, :]
        x = x + self.scale * pos_encoding
        
        # Expand temporal token for batch
        temp_tokens = self.temporal_token.expand(batch_size, -1, -1)
        
        # Concatenate temporal token with input
        x_with_temp = torch.cat([temp_tokens, x], dim=1)
        
        # Self-attention with temporal token
        attended, _ = self.temporal_attention(
            x_with_temp, x_with_temp, x_with_temp,
            need_weights=False
        )
        x_with_temp = x_with_temp + self.dropout(attended)
        x_with_temp = self.norm1(x_with_temp)
        
        # MLP block
        x_with_temp = x_with_temp + self.dropout(self.temporal_mlp(x_with_temp))
        x_with_temp = self.norm2(x_with_temp)
        
        # Remove temporal token
        x = x_with_temp[:, 1:, :]
        
        return self.dropout(x)

class TransformerEncoder(nn.Module):
	def __init__(self, num_heads, num_layers, d_model, output_dim, frames, dropout=0.1):
		super(TransformerEncoder, self).__init__()
		self.transformer_encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
		)
		self.transformer_encoder = nn.TransformerEncoder(
			self.transformer_encoder_layer, num_layers=num_layers
		)
		self.out = nn.Linear(d_model, output_dim)
		self.pos_encoder = PositionalEncoding(d_model, dropout)
		self.dropout = nn.Dropout(dropout)
		self.batch_norm = nn.BatchNorm1d(frames)
		torch.nn.init.kaiming_uniform_(self.out.weight, nonlinearity='relu')

	def forward(self, x):
		output = self.batch_norm(x)
		output = self.dropout(output)
		output = self.pos_encoder(output)
		output = self.transformer_encoder(output)
		output = self.out(output)

		return output

class MultiHeadClassifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super(MultiHeadClassifier, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, num_frames, d_model = x.shape
        out = self.fc(x.reshape(batch_size, num_frames * d_model))
        return out

class HybridCNNTransformerModel(nn.Module):
	def __init__(
		self,
		num_classes,
		feature_dim=2048,
		frame_samples=16,
		transformer_heads=8,
		transformer_layers=4,
		transformer_outputs=64,
		dropout=0.1
	):
		super(HybridCNNTransformerModel, self).__init__()
		self.cnn_feature_extractor = CNNFeatureExtractor()
		self.transformer_encoder = TransformerEncoder(
			output_dim=transformer_outputs, 
   			num_heads=transformer_heads,
      		num_layers=transformer_layers,
        	d_model=feature_dim,
			frames=frame_samples,
         	dropout=dropout
		)
		self.classifier = MultiHeadClassifier(frame_samples * transformer_outputs, num_classes)

	def forward(self, x):
		features = self.cnn_feature_extractor(x)
		transformer_output = self.transformer_encoder(features)
		out = self.classifier(transformer_output)
		return out